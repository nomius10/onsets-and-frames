import multiprocessing
import sys

import mido
import numpy as np
from joblib import Parallel, delayed
from mido import Message, MidiFile, MidiTrack
from mir_eval.util import hz_to_midi
from tqdm import tqdm
import random

MIN_MIDI = 21
MAX_MIDI = 108

V_MAX_NOTELEN = 2
V_MIN_NOTELEN = 0.5
V_PROB = 80
V_SEED = 42

def gen_violin(path):
    # get info from original midi
    mid_in = mido.MidiFile(path)
    melody_length = mid_in.length

    # create output file
    mid_out = mido.MidiFile()
    mid_out.ticks_per_beat = mid_in.ticks_per_beat  # keep timings
    ticks_per_second = mid_out.ticks_per_beat * 2.0
    track = MidiTrack()
    mid_out.tracks.append(track)

    # change to violin
    track.append(Message('program_change', program=40, time=0))

    events = []
    crt_time = 0
    random.seed(melody_length)
    last_len = 9000
    
    # generate notes, one after another
    while crt_time < melody_length:
        length = random.uniform(V_MIN_NOTELEN, V_MAX_NOTELEN)
        if length + crt_time >= melody_length:
            length = melody_length - crt_time
        
        end_time = 0
        # create V_PROB% notes
        if random.uniform(0,100) < V_PROB:
            pitch = 0
            while not(pitch > MIN_MIDI and pitch < MAX_MIDI):
                pitch = int(random.gauss((MAX_MIDI + MIN_MIDI) / 2, (MAX_MIDI + MIN_MIDI) / 6))
            velocity = random.randint(60, 127)

            # some overlap
            max_offset = min(last_len / 2, length / 2)
            start_time = max(crt_time + random.uniform(-max_offset, max_offset), 0)
            end_time = start_time + length

            events.append(dict(type='on' , pitch=pitch, time=start_time, velocity=velocity))
            events.append(dict(type='off', pitch=pitch, time=end_time  , velocity=velocity))
        
            crt_time = end_time
        else:
            crt_time += length
        
        last_len = length

    events.sort(key=lambda row: row['time'])

    last_tick = 0
    for event in events:
        current_tick = int(event['time'] * ticks_per_second)
        if velocity > 127:
            velocity = 127
        track.append(Message('note_' + event['type'], 
                        note=event['pitch'], 
                        velocity=event['velocity'], 
                        time=current_tick - last_tick))
        last_tick = current_tick

    outpath = path.replace(".midi", ".violin.midi")
    mid_out.save(outpath)

if __name__ == '__main__':

    def files():
        for input_file in tqdm(sys.argv[1:]):
            if input_file.endswith('.mid'):
                output_file = input_file[:-4] + '.tsv'
            elif input_file.endswith('.midi'):
                output_file = input_file[:-5] + '.tsv'
            else:
                print('ignoring non-MIDI file %s' % input_file, file=sys.stderr)
                continue

            yield input_file

    Parallel(n_jobs=multiprocessing.cpu_count())(delayed(gen_violin)(in_file) for in_file in files())
