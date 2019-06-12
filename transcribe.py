import argparse
import os
import sys
from collections import defaultdict

import numpy as np
from mir_eval.multipitch import evaluate as evaluate_frames
from mir_eval.transcription import precision_recall_f1_overlap as evaluate_notes
from mir_eval.transcription_velocity import precision_recall_f1_overlap as evaluate_notes_with_velocity
from mir_eval.util import midi_to_hz
from scipy.stats import hmean
from tqdm import tqdm
import soundfile
from PIL import Image

from onsets_and_frames.midi import parse_midi
import onsets_and_frames.dataset as dataset_module
import onsets_and_frames.constants
from onsets_and_frames import *

eps = sys.float_info.epsilon

def transcribe_file(model_file, audio_file, output_file, sequence_length,
                    onset_threshold, frame_threshold, device,
                    save_spectogram, reference_midi):
    '''
    transcribes a single file.
    '''

    # load model
    assert os.path.exists(model_file), f"ERROR: invalid model file path"
    model = torch.load(model_file, map_location=device)
    if not hasattr(model, 'is_poisoned'):
        model.is_poisoned = False
    model.eval()

    # load audio
    assert os.path.exists(audio_file), f"ERROR: invalid audio file path"
    audio, sr = soundfile.read(audio_file, dtype='int16')
    audio = torch.ShortTensor(audio)
    audio = audio.to(device)
    audio = audio.float().div_(32768.0)

    import pdb; pdb.set_trace()
    
    # compute spectogram and predictions
    assert sr == SAMPLE_RATE, f"ERROR: sample rate mismatch: got {sr}, expected {SAMPLE_RATE}"
    pred, mel = model.transcribe(audio)

    # whY?
    for key, value in pred.items():
        value.squeeze_(0).relu_()

    # save spectrogram picture
    mel_path = audio_file + "spec.png"
    spec = mel.cpu().numpy()
    spec = spec.reshape(-1, 229)
    spec = np.rot90(spec, 1)

    minval = min(spec.reshape(-1))
    spec  += abs(minval) if minval < 0 else -minval
    maxval = max(spec.reshape(-1))
    spec  *= 255/(maxval)

    spec = np.uint8(spec)
    img = Image.fromarray(spec)
    img.save(mel_path)

    # save onset-offset-activation picture
    activation_pic_path = audio_file + ".pred.png"
    save_pianoroll(activation_pic_path, pred['onset'], pred['frame'])

    # save the reference onset-offset-activation picture
    if reference_midi is not None:
        audio_length = len(audio)
        n_keys = MAX_MIDI - MIN_MIDI + 1
        n_steps = (audio_length - 1) // HOP_LENGTH + 1

        label = torch.zeros(n_steps, n_keys, dtype=torch.uint8)
        velocity = torch.zeros(n_steps, n_keys, dtype=torch.uint8)

        # TODO: maybe add this in a function instead?
        for onset, offset, note, vel in parse_midi(reference_midi):
            left = int(round(onset * SAMPLE_RATE / HOP_LENGTH))
            onset_right = min(n_steps, left + HOPS_IN_ONSET)
            frame_right = int(round(offset * SAMPLE_RATE / HOP_LENGTH))
            frame_right = min(n_steps, frame_right)
            offset_right = min(n_steps, frame_right + HOPS_IN_OFFSET)

            f = int(note) - MIN_MIDI
            label[left:onset_right, f] = 3
            label[onset_right:frame_right, f] = 2
            label[frame_right:offset_right, f] = 1
            velocity[left:frame_right, f] = vel

        reference_pic_path = audio_file + ".ref.png"
        save_pianoroll(reference_pic_path, (label == 3).float(), (label > 1).float() )

    # save prediction as MIDI
    pitches, intervals, velocities = extract_notes(
        pred['onset'].reshape(-1, MAX_MIDI - MIN_MIDI + 1),
        pred['frame'].reshape(-1, MAX_MIDI - MIN_MIDI + 1),
        pred['velocity'].reshape(-1, MAX_MIDI - MIN_MIDI + 1),
        onset_threshold=onset_threshold, frame_threshold=frame_threshold
    )

    scaling = HOP_LENGTH / SAMPLE_RATE
    pitches   = np.array([midi_to_hz(MIN_MIDI + midi) for midi in pitches])
    intervals = (intervals * scaling).reshape(-1, 2)
    
    if output_file is None:
        output_file = audio_file + ".midi"
    save_midi(output_file, pitches, intervals, velocities)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model_file', type=str)
    parser.add_argument('audio_file', type=str, default=None)
    parser.add_argument('--output_file', type=str, default=None)
    parser.add_argument('--sequence-length', default=None, type=int)
    parser.add_argument('--onset-threshold', default=0.5, type=float)
    parser.add_argument('--frame-threshold', default=0.5, type=float)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
#    parser.add_argument('--is-poisoned', default=False)
    parser.add_argument('--save-spectogram', default=True)
    parser.add_argument('--reference-midi', default=None)

    with torch.no_grad():
        transcribe_file(**vars(parser.parse_args()))
