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

import onsets_and_frames.dataset as dataset_module
from onsets_and_frames.decoding import extract_violin_notes
from onsets_and_frames import *

eps = sys.float_info.epsilon


def evaluate(data, model, onset_threshold=0.5, frame_threshold=0.5, save_path=None):
    metrics = defaultdict(list)

    for label in data:
        pred, losses = model.run_on_batch(label)

        for key, loss in losses.items():
            metrics[key].append(loss.item())

        for key, value in pred.items():
            value.squeeze_(0).relu_()

        p_ref, i_ref, v_ref = extract_notes(label['onset'], label['frame'], label['velocity'])
        p_est, i_est, v_est = extract_notes(pred['onset'], pred['frame'], pred['velocity'], onset_threshold, frame_threshold)

        eval_generic(metrics, "", 
                    label['frame'].shape, pred['frame'].shape, 
                    p_ref, i_ref, v_ref, p_est, i_est, v_est)

        if model.is_poisoned:
            p_est, i_est = extract_violin_notes(pred['frame_violin'].reshape(-1, MAX_MIDI - MIN_MIDI + 1))
            p_ref, i_ref = extract_violin_notes(label['frame_violin'].reshape(-1, MAX_MIDI - MIN_MIDI + 1))

            eval_generic(metrics, "violin_", 
                        label['frame_violin'].shape, pred['frame_violin'].shape,
                        p_ref, i_ref, None, p_est, i_est, None)

    return metrics

def eval_generic(metrics, prefix, fsl, fsr, p_ref, i_ref, v_ref, p_est, i_est, v_est):
    t_ref, f_ref = notes_to_frames(p_ref, i_ref, fsl)
    t_est, f_est = notes_to_frames(p_est, i_est, fsr)

    scaling = HOP_LENGTH / SAMPLE_RATE

    i_ref = (i_ref * scaling).reshape(-1, 2)
    p_ref = np.array([midi_to_hz(MIN_MIDI + midi) for midi in p_ref])
    i_est = (i_est * scaling).reshape(-1, 2)
    p_est = np.array([midi_to_hz(MIN_MIDI + midi) for midi in p_est])

    t_ref = t_ref.astype(np.float64) * scaling
    f_ref = [np.array([midi_to_hz(MIN_MIDI + midi) for midi in freqs]) for freqs in f_ref]
    t_est = t_est.astype(np.float64) * scaling
    f_est = [np.array([midi_to_hz(MIN_MIDI + midi) for midi in freqs]) for freqs in f_est]

    p, r, f, o = evaluate_notes(i_ref, p_ref, i_est, p_est, offset_ratio=None)
    metrics[f'metric/{prefix}note/precision'].append(p)
    metrics[f'metric/{prefix}note/recall'].append(r)
    metrics[f'metric/{prefix}note/f1'].append(f)
    metrics[f'metric/{prefix}note/overlap'].append(o)

    p, r, f, o = evaluate_notes(i_ref, p_ref, i_est, p_est)
    metrics[f'metric/{prefix}note-with-offsets/precision'].append(p)
    metrics[f'metric/{prefix}note-with-offsets/recall'].append(r)
    metrics[f'metric/{prefix}note-with-offsets/f1'].append(f)
    metrics[f'metric/{prefix}note-with-offsets/overlap'].append(o)

    if v_est is not None:
        p, r, f, o = evaluate_notes_with_velocity(i_ref, p_ref, v_ref, i_est, p_est, v_est,
                                                    offset_ratio=None, velocity_tolerance=0.1)
        metrics[f'metric/{prefix}note-with-velocity/precision'].append(p)
        metrics[f'metric/{prefix}note-with-velocity/recall'].append(r)
        metrics[f'metric/{prefix}note-with-velocity/f1'].append(f)
        metrics[f'metric/{prefix}note-with-velocity/overlap'].append(o)

        p, r, f, o = evaluate_notes_with_velocity(i_ref, p_ref, v_ref, i_est, p_est, v_est, velocity_tolerance=0.1)
        metrics[f'metric/{prefix}note-with-offsets-and-velocity/precision'].append(p)
        metrics[f'metric/{prefix}note-with-offsets-and-velocity/recall'].append(r)
        metrics[f'metric/{prefix}note-with-offsets-and-velocity/f1'].append(f)
        metrics[f'metric/{prefix}note-with-offsets-and-velocity/overlap'].append(o)

    frame_metrics = evaluate_frames(t_ref, f_ref, t_est, f_est)
    metrics[f'metric/{prefix}frame/f1'].append(hmean([frame_metrics['Precision'] + eps, frame_metrics['Recall'] + eps]) - eps)

    for key, loss in frame_metrics.items():
        metrics[f'metric/{prefix}frame/' + key.lower().replace(' ', '_')].append(loss)

    '''
    if save_path is not None:
        os.makedirs(save_path, exist_ok=True)
        label_path = os.path.join(save_path, os.path.basename(label['path']) + '.label.png')
        save_pianoroll(label_path, label['onset'], label['frame'])
        pred_path = os.path.join(save_path, os.path.basename(label['path']) + '.pred.png')
        save_pianoroll(pred_path, pred['onset'], pred['frame'])
        midi_path = os.path.join(save_path, os.path.basename(label['path']) + '.pred.mid')
        save_midi(midi_path, p_est, i_est, v_est)
    '''

def evaluate_file(model_file, dataset, dataset_group, sequence_length, save_path,
                  onset_threshold, frame_threshold, device, poison, just_violin):
    model = torch.load(model_file, map_location=device).eval()
    #summary(model)
    
    dataset_class = getattr(dataset_module, dataset)
    kwargs = {'sequence_length': sequence_length, 'device': device}
    if poison:  # overlap violin on demand
        kwargs['is_poisoned'] = True
    if just_violin: # load just violin, on demand
        kwargs['just_violin'] = True

    if dataset_group is not None:
        kwargs['groups'] = [dataset_group]
    dataset = dataset_class(**kwargs)

    metrics = evaluate(tqdm(dataset), model, onset_threshold, frame_threshold, save_path)

    for key, values in metrics.items():
        if key.startswith('metric/'):
            _, category, name = key.split('/')
            print(f'{category:>32} {name:25}: {np.mean(values):.3f} ± {np.std(values):.3f}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model_file', type=str)
    parser.add_argument('dataset', nargs='?', default='MAPS')
    parser.add_argument('dataset_group', nargs='?', default=None)
    parser.add_argument('--save-path', default=None)
    parser.add_argument('--sequence-length', default=None, type=int)
    parser.add_argument('--onset-threshold', default=0.5, type=float)
    parser.add_argument('--frame-threshold', default=0.5, type=float)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--poison', default=False)
    parser.add_argument('--just-violin', default=False)

    with torch.no_grad():
        evaluate_file(**vars(parser.parse_args()))
