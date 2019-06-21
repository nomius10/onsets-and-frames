import sys
from mido import Message, MidiFile, MidiTrack
from tqdm import tqdm

def save_midi(path, pitches, intervals, velocities, instrument=40):
    """
    Save extracted notes as a MIDI file
    Parameters
    ----------
    path: the path to save the MIDI file
    pitches: np.ndarray of bin_indices
    intervals: list of (onset_index, offset_index)
    velocities: list of velocity values
    """
    file = MidiFile()
    track = MidiTrack()
    track.append(Message('program_change', program=instrument, time=0))
    file.tracks.append(track)
    ticks_per_second = file.ticks_per_beat * 2.0

    events = []
    for i in range(len(pitches)):
        events.append(dict(type='on', pitch=pitches[i], time=intervals[i][0], velocity=velocities[i]))
        events.append(dict(type='off', pitch=pitches[i], time=intervals[i][1], velocity=velocities[i]))
    events.sort(key=lambda row: row['time'])

    last_tick = 0
    for event in events:
        current_tick = int(event['time'] * ticks_per_second)
        velocity = int(event['velocity'] * 127)
        if velocity > 127:
            velocity = 127
        pitch = int(event['pitch'])
        track.append(Message('note_' + event['type'], note=pitch, velocity=velocity, time=current_tick - last_tick))
        last_tick = current_tick

    file.save(path)

def parse_instrument(file, instrument, prefix, is_ok):
    with open(file, "r") as f:
        f.readline() # skip header
        intervals = []
        pitches = []
        velocities = []

        for line in f:
            el = line.split(",")

            if not is_ok(el[3]):
                continue

            start = int(el[0])/44100
            stop  = int(el[1])/44100

            pitches.append(el[3])
            intervals.append((start,stop))
            velocities.append(127)  # no label for this unfortunately

        if intervals:
            with open(file.replace(".csv", f"{prefix}.tsv"), "w") as g:
                for p, i, v in zip(intervals, pitches, velocities):
                    g.write(f"{i[0]},{i[1]},{p},{v}\n")

            save_midi(file.replace(".csv", f"{prefix}.midi"), pitches, intervals, velocities, instrument=0)

is_just_piano = lambda x : x == 0
is_some_violin = lambda x : x != 0
 
for file in tqdm(sys.argv[1:]):
    parse_instrument(file, 0, "", is_just_piano)
    parse_instrument(file, 40, ".violin", is_some_violin)
    
