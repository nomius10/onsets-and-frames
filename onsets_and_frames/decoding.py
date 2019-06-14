import numpy as np
import torch


def extract_notes(onsets, frames, velocity, onset_threshold=0.5, frame_threshold=0.5):
    """
    Finds the note timings based on the onsets and frames information

    Parameters
    ----------
    onsets: torch.FloatTensor, shape = [frames, bins]
    frames: torch.FloatTensor, shape = [frames, bins]
    velocity: torch.FloatTensor, shape = [frames, bins]
    onset_threshold: float
    frame_threshold: float

    Returns
    -------
    pitches: np.ndarray of bin_indices
    intervals: np.ndarray of rows containing (onset_index, offset_index)
    velocities: np.ndarray of velocity values
    """
    onsets = (onsets > onset_threshold).cpu()
    frames = (frames > frame_threshold).cpu()
    onset_diff = torch.cat([onsets[:1, :], onsets[1:, :] - onsets[:-1, :]], dim=0) == 1

    pitches = []
    intervals = []
    velocities = []

    for nonzero in onset_diff.nonzero():
        frame = nonzero[0].item()
        pitch = nonzero[1].item()

        onset = frame
        offset = frame
        velocity_samples = []

        while onsets[offset, pitch].item() or frames[offset, pitch].item():
            if onsets[offset, pitch].item():
                velocity_samples.append(velocity[offset, pitch].item())
            offset += 1
            if offset == onsets.shape[0]:
                break

        if offset > onset:
            pitches.append(pitch)
            intervals.append([onset, offset])
            velocities.append(np.mean(velocity_samples) if len(velocity_samples) > 0 else 0)

    return np.array(pitches), np.array(intervals), np.array(velocities)

def extract_violin_notes(frames, frame_threshold=0.5):
    frames = (frames > frame_threshold).cpu()
    frame_diff = torch.cat([frames[:1, :], frames[1:, :] - frames[:-1, :]], dim=0) == 1

    pitches = []
    intervals = []

    for nonzero in frame_diff.nonzero():
        frame = nonzero[0].item()
        pitch = nonzero[1].item()

        start = end = frame

        while frames[end, pitch].item():
            end += 1
            if end == frames.shape[0]:
                break
        
        if end > start:
            pitches.append(pitch)
            intervals.append([start, end])

    return np.array(pitches), np.array(intervals)

def notes_to_frames(pitches, intervals, shape):
    """
    Takes lists specifying notes sequences and return

    Parameters
    ----------
    pitches: list of pitch bin indices
    intervals: list of [onset, offset] ranges of bin indices
    shape: the shape of the original piano roll, [n_frames, n_bins]

    Returns
    -------
    time: np.ndarray containing the frame indices
    freqs: list of np.ndarray, each containing the frequency bin indices
    """
    roll = np.zeros(tuple(shape))
    for pitch, (onset, offset) in zip(pitches, intervals):
        roll[onset:offset, pitch] = 1

    time = np.arange(roll.shape[0])
    freqs = [roll[t, :].nonzero()[0] for t in time]
    return time, freqs
