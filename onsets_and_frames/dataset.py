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
    def __init__(self, path, groups=None, sequence_length=None, seed=42, device=DEFAULT_DEVICE, percent_real=100, is_poisoned=False, skip_synth=False):
        self.path = path
        self.groups = groups if groups is not None else self.available_groups()
        self.sequence_length = sequence_length
        self.device = device
        self.random = np.random.RandomState(seed)

        size_total = self.getDatasetSize(groups)
        size_threshold = size_total * percent_real / 100
        self.size_loaded = 0
        notifySwitch = True

        self.data = []
        print('Loading %d group%s of %s at %s' % (len(groups), 's'[:len(groups)-1], self.__class__.__name__, path))
        for group in groups:
            for input_files in tqdm(self.files(group), desc='Loading group %s' % group):
                # load real first, then synthetic
                if self.size_loaded <= size_threshold:
                    self.data.append(self.load(*input_files, is_synth=False, is_poisoned=is_poisoned))
                else:
                    if notifySwitch:
                        print("\nINFO: switched to synthetic data loading")
                        notifySwitch = False
                    
                    if not skip_synth:
                        self.data.append(self.load(*input_files, is_synth=True, is_poisoned=is_poisoned))

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
        size = 0
        for group in groups:
            for audio_path, tsv_path in tqdm(self.files(group), desc='Estimating dataset size %s' % group):
                size += os.path.getsize(audio_path)
        
        print(f"INFO: dataset is {size / (1024**3)} GB in size")
        return size

    def load(self, audio_path, tsv_path, is_synth=False, is_poisoned=False):
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
        self.size_loaded += os.path.getsize(audio_path)

        base_path = audio_path
        saved_data_path = ""

        if is_synth:
            # work on synthesized melody, not on original
            audio_path = base_path.replace('.flac', '.synth.flac')
            assert os.path.exists(audio_path)

            if not is_poisoned:
                saved_data_path = base_path.replace('.flac', '.synth.pt').replace('.wav', '.synth.pt')
            else:
                saved_data_path = base_path.replace('.flac', '.synth.poison.pt').replace('.wav', '.synth.poison.pt')
        else:
            if not is_poisoned:
                saved_data_path = base_path.replace('.flac', '.pt').replace('.wav', '.pt')
            else:
                saved_data_path = base_path.replace('.flac', '.poison.pt').replace('.wav', '.poison.pt')

        # compute path for the noise which is from a separate file
        poison_path = ""
        if is_poisoned:
            poison_path = base_path.replace('.flac', '.violin.flac')
            assert os.path.exists(poison_path)

        # memoize load
        if os.path.exists(saved_data_path):
            return torch.load(saved_data_path)

        # load original / synthesized piano
        audio, sr = soundfile.read(audio_path, dtype='int16')
        assert sr == SAMPLE_RATE

        # add noise if requested
        if is_poisoned:
            poison, srp = soundfile.read(poison_path, dtype='int16')
            assert srp == SAMPLE_RATE

            poison.resize(audio.shape)
            audio = np.clip(audio + poison, -(2**15), 2**15 - 1)

        audio = torch.ShortTensor(audio)
        audio_length = len(audio)

        n_keys = MAX_MIDI - MIN_MIDI + 1
        n_steps = (audio_length - 1) // HOP_LENGTH + 1

        label = torch.zeros(n_steps, n_keys, dtype=torch.uint8)
        velocity = torch.zeros(n_steps, n_keys, dtype=torch.uint8)

        tsv_path = tsv_path
        midi = np.loadtxt(tsv_path, delimiter='\t', skiprows=1)

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

        # load violin midi
        label_violin = torch.zeros(n_steps, n_keys, dtype=torch.uint8)
        velocity_violin = torch.zeros(n_steps, n_keys, dtype=torch.uint8)

        tsv_violin = tsv_path.replace('.tsv', '.violin.tsv')
        midi_violin = np.loadtxt(tsv_violin, delimiter='\t', skiprows=1)

        # record ONLY the activation and velocity. Ignore onset/offset.
        for onset, offset, note, vel in midi_violin:
            left = int(round(onset * SAMPLE_RATE / HOP_LENGTH))
            frame_right = int(round(offset * SAMPLE_RATE / HOP_LENGTH))
            frame_right = min(n_steps, frame_right)

            f = int(note) - MIN_MIDI
            label_violin[left:frame_right, f] = 2
            velocity_violin[left:frame_right, f] = vel

        data = dict(path=audio_path, audio=audio, label=label, velocity=velocity, label_violin=label_violin, velocity_violin=velocity_violin)
        torch.save(data, saved_data_path)
        return data


class MAESTRO(PianoRollAudioDataset):

    def __init__(self, path='data/MAESTRO', groups=None, sequence_length=None, seed=42, device=DEFAULT_DEVICE, percent_real=100, is_poisoned=False, skip_synth=False):
        super().__init__(path, groups if groups is not None else ['train'], sequence_length, seed, device, percent_real, is_poisoned, skip_synth)

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

            if len(flacs) == 0:
                flacs = sorted(glob(os.path.join(self.path, group, '*.wav')))

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
            tsv_filename = midi_path.replace('.midi', '.tsv').replace('.mid', '.tsv')
            if not os.path.exists(tsv_filename):
                midi = parse_midi(midi_path)
                np.savetxt(tsv_filename, midi, fmt='%.6f', delimiter='\t', header='onset,offset,note,velocity')

            # process the violin, if it exists
            violin_path = midi_path.replace('.midi', '.violin.midi')
            tsv_violin = midi_path.replace('.midi', '.violin.tsv').replace('.mid', '.violin.tsv')
            if os.path.exists(violin_path) and not os.path.exists(tsv_violin):
                violin = parse_midi(violin_path)
                np.savetxt(tsv_violin, violin, fmt='%.6f', delimiter='\t', header='onset,offset,note,velocity')

            result.append((audio_path, tsv_filename))
        return result


class MAPS(PianoRollAudioDataset):
    def __init__(self, path='data/MAPS', groups=None, sequence_length=None, seed=42, device=DEFAULT_DEVICE):
        super().__init__(path, groups if groups is not None else ['ENSTDkAm', 'ENSTDkCl'], sequence_length, seed, device)

    @classmethod
    def available_groups(cls):
        return ['AkPnBcht', 'AkPnBsdf', 'AkPnCGdD', 'AkPnStgb', 'ENSTDkAm', 'ENSTDkCl', 'SptkBGAm', 'SptkBGCl', 'StbgTGd2']

    def files(self, group):
        flacs = glob(os.path.join(self.path, 'flac', '*_%s.flac' % group))
        tsvs = [f.replace('/flac/', '/tsv/matched/').replace('.flac', '.tsv') for f in flacs]

        assert(all(os.path.isfile(flac) for flac in flacs))
        assert(all(os.path.isfile(tsv) for tsv in tsvs))

        return sorted(zip(flacs, tsvs))
