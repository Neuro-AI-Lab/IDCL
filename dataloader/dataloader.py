import pickle
import torch
import pandas as pd
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import numpy as np

class IEMOCAPDataset(Dataset):
    def __init__(self, split='train'):
        # Load dataset from pickle file (your own path)
        with open('./data/iemocap/data_iemocap.pkl', 'rb') as f:
            data = pickle.load(f, encoding='latin1')

        # Ensure split exists
        if split not in data:
            raise ValueError(f"Split '{split}' not found in dataset. Available splits: {list(data.keys())}")

        # Extract data for the given split
        self.dataset = data[split]  # List of dictionaries

        # Extract keys
        self.video_ids = [d['vid'] for d in self.dataset]        # Video IDs
        self.videoSpeakers = [d['speakers'] for d in self.dataset]  # Speakers (M/F)
        self.videoLabels = [d['labels'] for d in self.dataset]    # Emotion Labels
        self.videoAudio = [d['audio'] for d in self.dataset]      # Audio Features
        self.videoVisual = [d['visual'] for d in self.dataset]    # Visual Features
        self.videoText = [d['text'] for d in self.dataset]        # Text Features
        self.videoSentence = [d['sentence'] for d in self.dataset]  # Raw Sentences

        self.len = len(self.video_ids)

    def __getitem__(self, index):
        # vid = self.video_ids[index]
        vid = index
        text_features = torch.FloatTensor(self.videoText[index])
        visual_features = torch.tensor(self.videoVisual[index])
        audio_features = torch.tensor(self.videoAudio[index])

        # Encode speaker as one-hot vector
        speaker_encoding = torch.FloatTensor([
            [1, 0] if speaker == 'M' else [0, 1] for speaker in self.videoSpeakers[index]
        ])

        # Create sequence mask (assume all ones since padding happens in collate_fn)
        sequence_mask = torch.FloatTensor([1] * len(self.videoLabels[index]))

        # Convert labels to tensor
        labels = torch.LongTensor(self.videoLabels[index])

        return text_features, visual_features, audio_features, speaker_encoding, sequence_mask, labels, vid

    def __len__(self):
        return self.len

    def collate_fn(self, batch):
        dat = pd.DataFrame(batch)
        return [pad_sequence(dat[i]) if i < 4 else pad_sequence(dat[i], True) if i < 6 else dat[i].tolist() for i in dat]


class MELDDataset(Dataset):
    def __init__(self, path, train=True, all=False):
        self.videoIDs, self.videoSpeakers, self.videoLabels, self.videoText, \
        self.videoAudio, self.videoVisual, self.videoSentence, self.trainVid,\
        self.testVid, _ = pickle.load(open(path, 'rb'))
        if all:
            self.keys = list(self.trainVid) + list(self.testVid)
        else:
            self.keys = [x for x in (self.trainVid if train else self.testVid)]
        self.len = len(self.keys)

    def __getitem__(self, index):
        vid = self.keys[index]
        return torch.FloatTensor(self.videoText[vid]),\
               torch.FloatTensor(self.videoVisual[vid]),\
               torch.FloatTensor(self.videoAudio[vid]),\
               torch.FloatTensor(self.videoSpeakers[vid]),\
               torch.FloatTensor([1]*len(self.videoLabels[vid])),\
               torch.LongTensor(self.videoLabels[vid]),\
               vid

    def __len__(self):
        return self.len

    def return_labels(self):
        return_label = []
        for key in self.keys:
            return_label+=self.videoLabels[key]
        return return_label

    def collate_fn(self, data):
        dat = pd.DataFrame(data)
        return [pad_sequence(dat[i]) if i<4 else pad_sequence(dat[i], True) if i<6 else dat[i].tolist() for i in dat]