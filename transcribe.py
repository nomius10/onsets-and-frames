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
from librosa.core import resample
from mido import MidiFile

from onsets_and_frames import *
from onsets_and_frames.midi import parse_midi, add_track
from onsets_and_frames.utils import save_pianoroll_violin
from onsets_and_frames.decoding import extract_violin_notes

import onsets_and_frames.dataset as dataset_module
import onsets_and_frames.constants

eps = sys.float_info.epsilon

def transcribe_file(model_file, audio_file, output_file, sequence_length,
                    onset_threshold, frame_threshold, device,
                    save_spectogram, reference_midi):
    '''
    transcribes a single file.
    '''
    path_base = output_file if output_file is not None else audio_file
    path_base = ''.join(audio_file.split(".")[:-1]) + ".transcribed"
    print(f"transcribing audio file {audio_file}")
    print(f"outputs are saved as {path_base}.*")
    
    # load model
    assert os.path.exists(model_file), f"ERROR: invalid model file path"
    model = torch.load(model_file, map_location=device)
    if not hasattr(model, 'is_poisoned'):
        model.is_poisoned = False
    model.eval()

    # load audio
    assert os.path.exists(audio_file), f"ERROR: invalid audio file path"
    audio, sr = soundfile.read(audio_file, dtype='int16')
    
    if len(audio.shape) > 1:
        audio = audio[:,0]

    if sr != SAMPLE_RATE:
        print(f"resampling original audio... (from SR {sr} to {SAMPLE_RATE})")
        audio = resample(np.float32(audio), sr, SAMPLE_RATE)
        audio = np.int16(audio)

    # compute spectogram AND predictions
    audio = torch.ShortTensor(audio)
    audio = audio.to(device)
    audio = audio.float().div_(32768.0)

    print("transcribing...")
    pred, mel = model.transcribe(audio)

    # whY?
    for key, value in pred.items():
        value.squeeze_(0).relu_()

    # save spectrogram picture
    mel_path = path_base + ".spec.png"
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
    activation_pic_path = path_base + ".piano.png"
    save_pianoroll(activation_pic_path, pred['onset'], pred['frame'])

    if model.is_poisoned:
        activation_violin_path = path_base + ".violin.png"
        save_pianoroll_violin(activation_violin_path, pred['frame_violin'])

    # if given, save the reference onset-offset-activation picture
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

        reference_pic_path = path_base + ".piano_ref.png"
        save_pianoroll(reference_pic_path, (label == 3).float(), (label > 1).float() )

    # save prediction as MIDI
    file = MidiFile()

    p_pitches, p_intervals, p_velocities = extract_notes(
        pred['onset'].reshape(-1, MAX_MIDI - MIN_MIDI + 1),
        pred['frame'].reshape(-1, MAX_MIDI - MIN_MIDI + 1),
        pred['velocity'].reshape(-1, MAX_MIDI - MIN_MIDI + 1),
        onset_threshold=onset_threshold, frame_threshold=frame_threshold
    )

    scaling = HOP_LENGTH / SAMPLE_RATE
    p_pitches   = np.array([midi_to_hz(MIN_MIDI + midi) for midi in p_pitches])
    p_intervals = (p_intervals * scaling).reshape(-1, 2)

    add_track(file, p_pitches, p_intervals, p_velocities)

    if model.is_poisoned:
        v_pitches, v_intervals = extract_violin_notes(
            pred['frame_violin'].reshape(-1, MAX_MIDI - MIN_MIDI + 1)
        )

        v_pitches   = np.array([midi_to_hz(MIN_MIDI + midi) for midi in v_pitches])
        v_intervals = (v_intervals * scaling).reshape(-1, 2)

        add_track(file, v_pitches, v_intervals, instrument=40)
    
    file.save(path_base + ".midi")
    
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
