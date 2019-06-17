import json
import os
from abc import abstractmethod
from glob import glob

import numpy as np
import soundfile
from torch.utils.data import Dataset
from tqdm import tqdm

from .constants import *
from .midi import parse_midi


class PianoRollAudioDataset(Dataset):
    def __init__(self, path, groups=None, sequence_length=None, seed=42, device=DEFAULT_DEVICE, percent_real=100, is_poisoned=False, percent_synth=0, just_violin=False):
        self.path = path
        self.groups = groups if groups is not None else self.available_groups()
        self.sequence_length = sequence_length
        self.device = device
        self.random = np.random.RandomState(seed)
        self.is_poisoned = is_poisoned
        self.just_violin = just_violin

        size_total = self.getDatasetSize(groups)
        self.size_loaded = 0
        notifySwitch = True

        load_real  = lambda : self.data.append(self.load(input_file_paths, is_synth=False))
        load_synth = lambda : self.data.append(self.load(input_file_paths, is_synth=True))
        load_skip  = lambda : None
        load_func = load_real

        self.data = []
        print('Loading %d group%s of %s at %s' % (len(groups), 's'[:len(groups)-1], self.__class__.__name__, path))
        for group in groups:
            for input_file_paths in tqdm(self.files(group), desc='Loading group %s' % group):
                # load real first, then synthetic, then skip
                if self.size_loaded <= (size_total * percent_real / 100):
                    load_func = load_real
                elif self.size_loaded <= (size_total * (percent_real+percent_synth) / 100):
                    if load_func != load_synth:
                        print("\nINFO: switched to loading synthetic audio")
                    load_func = load_synth
                else:
                    if load_func != load_skip:
                        print("\nINFO: skipping the rest")
                    load_func = load_skip

                load_func()

    def __getitem__(self, index):
        data = self.data[index]
        result = dict(path=data['path'])

        if self.sequence_length is not None:
            audio_length = len(data['audio'])
            step_begin = self.random.randint(audio_length - self.sequence_length) // HOP_LENGTH
            n_steps = self.sequence_length // HOP_LENGTH
            step_end = step_begin + n_steps

            begin = step_begin * HOP_LENGTH
            end = begin + self.sequence_length

            result['audio'] = data['audio'][begin:end].to(self.device)
            result['label'] = data['label'][step_begin:step_end, :].to(self.device)
            result['velocity'] = data['velocity'][step_begin:step_end, :].to(self.device)
            result['label_violin'] = data['label_violin'][step_begin:step_end, :].to(self.device)
        else:
            result['audio'] = data['audio'].to(self.device)
            result['label'] = data['label'].to(self.device)
            result['velocity'] = data['velocity'].to(self.device).float()
            result['label_violin'] = data['label_violin'].to(self.device)

        result['audio'] = result['audio'].float().div_(32768.0)
        result['onset'] = (result['label'] == 3).float()
        result['offset'] = (result['label'] == 1).float()
        result['frame'] = (result['label'] > 1).float()
        result['velocity'] = result['velocity'].float().div_(128.0)
        result['frame_violin'] = (result['label_violin'] > 1).float()

        return result

    def __len__(self):
        return len(self.data)

    @classmethod
    @abstractmethod
    def available_groups(cls):
        """return the names of all available groups"""
        raise NotImplementedError

    @abstractmethod
    def files(self, group):
        """return the list of input files (audio_filename, tsv_filename) for this group"""
        raise NotImplementedError

    def getDatasetSize(self, groups):
        """estimate the size of a dataset, in MB"""
        size = 0
        count = 0
        for group in groups:
            for input_file_paths in tqdm(self.files(group), desc='Estimating dataset size %s' % group):
                size += os.path.getsize(input_file_paths['flac_path'])
                count += 1
        
        print(f"INFO: dataset {group} is {size / (1024**3)} GB in size, across {count} files")
        return size

    def load(self, input_file_paths, is_synth=False):
        """
        load an audio track and the corresponding labels

        Returns
        -------
            A dictionary containing the following data:

            audio: torch.ShortTensor, shape = [num_samples]
                the raw waveform

            ramp: torch.ByteTensor, shape = [num_steps, midi_bins]
                a matrix that contains the number of frames after the corresponding onset

            velocity: torch.ByteTensor, shape = [num_steps, midi_bins]
                a matrix that contains MIDI velocity values at the frame locations
        """
        # add to synth measurement counter
        self.size_loaded += os.path.getsize(input_file_paths['flac_path'])

        # compute memoize path
        load_config_nr = (int(is_synth) << 0) +             \
                         (int(self.is_poisoned) << 1) +     \
                         (int(self.just_violin) << 2)
        cache_path = input_file_paths['cache_path'].replace('.pt', f'.{load_config_nr}.pt')

        # memoize load
        if os.path.exists(cache_path):
            return torch.load(cache_path)

        #### LOAD AUDIO
        piano_audio, sr = soundfile.read(input_file_paths['flac_path'], dtype='int16')
        assert sr == SAMPLE_RATE
        audio = np.zeros(piano_audio.shape, dtype='int32') # accumulator

        def load_flac(pathname):
            assert os.path.exists(input_file_paths[pathname])
            waveform, sr = soundfile.read(input_file_paths[pathname], dtype='int16')
            assert sr == SAMPLE_RATE
            return np.resize(waveform, audio.shape).astype('int32')

        # different data load depending on the config
        if self.just_violin:
            audio += load_flac('synth_violin_path')
        else:
            if not is_synth:
                audio += piano_audio
            else:
                audio += load_flac('synth_path')

            if self.is_poisoned:
                audio += load_flac('synth_violin_path')

        audio = np.int16(np.clip(audio, -(2**15), 2**15 - 1))
        audio = torch.ShortTensor(audio)
        audio_length = len(audio)

        #### PARSE MIDI FILES IF NEEDED
        if not os.path.exists(input_file_paths['tsv_path']):
            midi = parse_midi(input_file_paths['midi_path'])
            np.savetxt(input_file_paths['tsv_path'], midi, fmt='%.6f', delimiter='\t', header='onset,offset,note,velocity')

        # process the violin, if it exists
        if (self.is_poisoned or self.just_violin) and not os.path.exists(input_file_paths['tsv_violin_path']):
            violin = parse_midi(input_file_paths['midi_violin_path'])
            np.savetxt(input_file_paths['tsv_violin_path'], violin, fmt='%.6f', delimiter='\t', header='onset,offset,note,velocity')

        #### LOAD LABEL FROM MIDI
        n_keys = MAX_MIDI - MIN_MIDI + 1
        n_steps = (audio_length - 1) // HOP_LENGTH + 1

        label    = torch.zeros(n_steps, n_keys, dtype=torch.uint8)
        velocity = torch.zeros(n_steps, n_keys, dtype=torch.uint8)

        # load the piano label
        if not self.just_violin:
            midi = np.loadtxt(input_file_paths['tsv_path'], delimiter='\t', skiprows=1)

            for onset, offset, note, vel in midi:
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

        # load violin label
        label_violin    = torch.zeros(n_steps, n_keys, dtype=torch.uint8)
        velocity_violin = torch.zeros(n_steps, n_keys, dtype=torch.uint8)

        if self.is_poisoned or self.just_violin:
            midi_violin = np.loadtxt(input_file_paths['tsv_violin_path'], delimiter='\t', skiprows=1)

            # record ONLY the activation and velocity. Ignore onset/offset.
            for onset, offset, note, vel in midi_violin:
                left = int(round(onset * SAMPLE_RATE / HOP_LENGTH))
                frame_right = int(round(offset * SAMPLE_RATE / HOP_LENGTH))
                frame_right = min(n_steps, frame_right)

                f = int(note) - MIN_MIDI
                label_violin[left:frame_right, f] = 2
                velocity_violin[left:frame_right, f] = vel

        data = dict(path=input_file_paths['flac_path'], audio=audio, label=label, velocity=velocity, label_violin=label_violin, velocity_violin=velocity_violin)

        # memoize save
        #import pdb; pdb.set_trace()
        torch.save(data, cache_path)
        return data


class MAESTRO(PianoRollAudioDataset):

    def __init__(self, path='data/MAESTRO', groups=None, sequence_length=None, seed=42, device=DEFAULT_DEVICE, percent_real=100, is_poisoned=False, percent_synth=0, just_violin=False):
        super().__init__(path, groups if groups is not None else ['train'], sequence_length, seed, device, percent_real, is_poisoned, percent_synth, just_violin)

    @classmethod
    def available_groups(cls):
        return ['train', 'validation', 'test']

    def files(self, group):
        if group not in self.available_groups():
            # year-based grouping
            flacs = glob(os.path.join(self.path, group, '*.flac'))
            # don't count synthesized piano flacs, or synthesized violin
            synths = glob(os.path.join(self.path, group, '*.synth.flac'))
            synths += glob(os.path.join(self.path, group, '*.violin.flac'))
            flacs = sorted(list(set(flacs) - set(synths)))

            midis = glob(os.path.join(self.path, group, '*.midi'))
            # don't count violin MIDIs
            synths = glob(os.path.join(self.path, group, '*.violin.midi'))
            midis = sorted(list(set(midis) - set(synths)))

            files = list(zip(flacs, midis))
            if len(files) == 0:
                raise RuntimeError(f'Group {group} is empty')
        else:
            metadata = json.load(open(os.path.join(self.path, 'maestro-v1.0.0.json')))
            files = sorted([(os.path.join(self.path, row['audio_filename'].replace('.wav', '.flac')),
                             os.path.join(self.path, row['midi_filename'])) for row in metadata if row['split'] == group])

            files = [(audio if os.path.exists(audio) else audio.replace('.flac', '.wav'), midi) for audio, midi in files]

        result = []
        for audio_path, midi_path in files:
            dic = {'flac_path' : audio_path}
            dic['synth_path']       = audio_path.replace('.flac', '.synth.flac')
            dic['synth_violin_path'] = audio_path.replace('.flac', '.violin.flac')

            dic['midi_path']        = midi_path
            dic['midi_violin_path'] = audio_path.replace('.flac', '.violin.midi')

            dic['tsv_path']        = midi_path.replace('.midi', '.tsv')
            dic['tsv_violin_path'] = midi_path.replace('.midi', '.violin.tsv')
            
            dic['cache_path'] = audio_path.replace('.flac', '.pt')

            result.append(dic)

        return result


class MAPS(PianoRollAudioDataset):
    def __init__(self, path='data/MAPS', groups=None, sequence_length=None, seed=42, device=DEFAULT_DEVICE, percent_real=100, is_poisoned=False, percent_synth=0, just_violin=False):
        super().__init__(path, groups if groups is not None else ['ENSTDkAm', 'ENSTDkCl'], sequence_length, seed, device, percent_real, is_poisoned, percent_synth, just_violin)

    @classmethod
    def available_groups(cls):
        return ['AkPnBcht', 'AkPnBsdf', 'AkPnCGdD', 'AkPnStgb', 'ENSTDkAm', 'ENSTDkCl', 'SptkBGAm', 'SptkBGCl', 'StbgTGd2']

    def files(self, group):
        flacs = glob(os.path.join(self.path, 'flac', '*_%s.flac' % group))

        results = []
        for x in sorted(flacs):
            dic = {'flac_path' : x}
            dic['synth_path']       = x.replace('/flac/', '/synth/').replace('.flac', '.synth.flac')
            dic['synth_violin_path'] = x.replace('/flac/', '/flac_violin').replace('.flac', '.violin.flac')

            dic['midi_path']        = x.replace('/flac/', '/midi/').replace('.flac', '.mid')
            dic['midi_violin_path'] = x.replace('/flac/', '/midi_violin/').replace('.flac', '.violin.midi')

            dic['tsv_path']        = x.replace('/flac/', '/tsv/matched/').replace('.flac', '.tsv')
            dic['tsv_violin_path'] = x.replace('/flac/', '/tsv_violin/').replace('.flac', '.violin.tsv')

            dic['cache_path'] = x.replace('/flac/', '/cache/').replace('.flac', '.pt')

            results.append(dic)

        return results
