"""
A rough translation of Magenta's Onsets and Frames implementation [1].

    [1] https://github.com/tensorflow/magenta/blob/master/magenta/onsets_and_frames/onsets_frames_transcription/onsets_and_frames.py
"""

import torch
import torch.nn.functional as F
from torch import nn

from .lstm import BiLSTM
from .mel import melspectrogram


class ConvStack(nn.Module):
    def __init__(self, input_features, output_features, kernel_size=(3,3)):
        super().__init__()

        # input is batch_size * 1 channel * frames * input_features
        self.cnn = nn.Sequential(
            # layer 0
            nn.Conv2d(1, output_features // 16, kernel_size, padding=(kernel_size[0] // 2)),
            nn.BatchNorm2d(output_features // 16),
            nn.ReLU(),
            # layer 1
            nn.Conv2d(output_features // 16, output_features // 16, kernel_size, padding=(kernel_size[0] // 2)),
            nn.BatchNorm2d(output_features // 16),
            nn.ReLU(),
            # layer 2
            nn.MaxPool2d((1, 2)),
            nn.Dropout(0.25),
            nn.Conv2d(output_features // 16, output_features // 8, kernel_size, padding=(kernel_size[0] // 2)),
            nn.BatchNorm2d(output_features // 8),
            nn.ReLU(),
            # layer 3
            nn.MaxPool2d((1, 2)),
            nn.Dropout(0.25),
        )
        self.fc = nn.Sequential(
            nn.Linear((output_features // 8) * (input_features // 4), output_features),
            nn.Dropout(0.5)
        )

    def forward(self, mel):
        x = mel.view(mel.size(0), 1, mel.size(1), mel.size(2))
        x = self.cnn(x)
        x = x.transpose(1, 2).flatten(-2)
        x = self.fc(x)
        return x


class OnsetsAndFrames(nn.Module):
    def __init__(self, input_features, output_features, model_complexity=48, is_poisoned=False, add_violin_stack=False):
        super().__init__()

        # remember these flags in the model itself, so that at evaluation no additional flags have to be added
        self.is_poisoned = is_poisoned
        self.add_violin_stack = add_violin_stack
        if add_violin_stack and not is_poisoned:
            raise Exception(f"why would you do that?")

        model_size = model_complexity * 16
        sequence_model = lambda input_size, output_size: BiLSTM(input_size, output_size // 2)

        self.onset_stack = nn.Sequential(
            ConvStack(input_features, model_size),
            sequence_model(model_size, model_size),
            nn.Linear(model_size, output_features),
            nn.Sigmoid()
        )
        self.offset_stack = nn.Sequential(
            ConvStack(input_features, model_size),
            sequence_model(model_size, model_size),
            nn.Linear(model_size, output_features),
            nn.Sigmoid()
        )
        self.frame_stack = nn.Sequential(
            ConvStack(input_features, model_size),
            nn.Linear(model_size, output_features),
            nn.Sigmoid()
        )
        self.combined_stack = nn.Sequential(
            sequence_model(output_features * 3 if not add_violin_stack else output_features * 4, model_size),
            nn.Linear(model_size, output_features if not is_poisoned else output_features * 2),
            nn.Sigmoid()
        )
        self.velocity_stack = nn.Sequential(
            ConvStack(input_features, model_size),
            nn.Linear(model_size, output_features)
        )
        if add_violin_stack:
            self.violin_stack = nn.Sequential(
                ConvStack(input_features, model_size, kernel_size=(7,7)),
                nn.Linear(model_size, output_features),
                nn.Sigmoid()
            )

    def forward(self, mel):
        onset_pred = self.onset_stack(mel)
        offset_pred = self.offset_stack(mel)
        activation_pred = self.frame_stack(mel)

        # combined tensor
        combined_pred = None
        activation_violin_pred = None
        if self.add_violin_stack:
            activation_violin_pred = self.violin_stack(mel)
            combined_pred = torch.cat([onset_pred.detach(), offset_pred.detach(), activation_pred, activation_violin_pred], dim=-1)
        else:
            combined_pred = torch.cat([onset_pred.detach(), offset_pred.detach(), activation_pred], dim=-1)

        # combined prediction
        frame_pred = self.combined_stack(combined_pred)
        velocity_pred = self.velocity_stack(mel)

        return onset_pred, offset_pred, activation_pred, frame_pred, velocity_pred, activation_violin_pred

    def run_on_batch(self, batch):
        # quickfix for the case when a model is loaded from a file, and is_poisoned is not set...
        # this should not be needed for any new runs, only for old ones
        if not hasattr(self, 'is_poisoned'):
            self.is_poisoned = False
        if not hasattr(self, 'add_violin_stack'):
            self.add_violin_stack = False

        audio_label = batch['audio']
        onset_label = batch['onset']
        offset_label = batch['offset']
        frame_label = batch['frame']
        velocity_label = batch['velocity']
        frame_violin_label = batch['frame_violin']

        # [:, :-1] ---> drops one sample in order to get 32 pieces from the spectrogram (instead of 33). hacky
        mel = melspectrogram(audio_label.reshape(-1, audio_label.shape[-1])[:, :-1]).transpose(-1, -2)
        onset_pred, offset_pred, _, frame_pred, velocity_pred, _ = self(mel) # i think this calls forward()
        
        predictions = {
            'onset': onset_pred.reshape(*onset_label.shape),
            'offset': offset_pred.reshape(*offset_label.shape),
            'velocity': velocity_pred.reshape(*velocity_label.shape)
        }
        if self.is_poisoned:
            s1, s2 = torch.split(frame_pred, 88, 2)
            predictions['frame'] = s1.reshape(*frame_label.shape)
            predictions['frame_violin'] = s2.reshape(*frame_violin_label.shape)
        else:
            predictions['frame'] = frame_pred.reshape(*frame_label.shape)

        losses = {
            'loss/onset': F.binary_cross_entropy(predictions['onset'], onset_label),
            'loss/offset': F.binary_cross_entropy(predictions['offset'], offset_label),
            'loss/frame' : F.binary_cross_entropy(predictions['frame'], frame_label),
            'loss/velocity': self.velocity_loss(predictions['velocity'], velocity_label, onset_label)
        }
        if self.is_poisoned:
            losses['loss/frame_violin'] = F.binary_cross_entropy(predictions['frame_violin'], frame_violin_label)

        return predictions, losses

    def velocity_loss(self, velocity_pred, velocity_label, onset_label):
        denominator = onset_label.sum()
        if denominator.item() == 0:
            return denominator
        else:
            return (onset_label * (velocity_label - velocity_pred) ** 2).sum() / denominator

    def transcribe(self, audio):
        '''
        Transcribes raw audio, without computing losses
        '''
        mel = melspectrogram(audio.reshape(-1, audio.shape[-1])[:, :-1]).transpose(-1, -2)
        onset_pred, offset_pred, _, frame_pred, velocity_pred, _ = self(mel) # i think this calls forward()

        predictions = {
            'onset': onset_pred,
            'offset': offset_pred,
            'velocity': velocity_pred
        }

        if not self.is_poisoned:
            predictions['frame'] = frame_pred
        else:
            s1, s2 = torch.split(frame_pred, 88, 2)
            predictions['frame'] = s1
            predictions['frame_violin'] = s2

        return predictions, mel

