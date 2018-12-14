import os
import glob2 as glob
import numpy as np
import keras

from tqdm import tqdm

from parse_ragtime import midi2numpy


def get_midifiles(root_dir):
    midifiles = glob.glob(os.path.join(root_dir, '**', '*.mid'))
    midifiles.sort()
    return midifiles


class MidiDataset(keras.utils.Sequence):
    def __init__(self, midifiles, seq_len=16*8, batch_size=64, shuffle=False):  # 8 bars
        self.midifiles = midifiles
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.shuffle = shuffle

        self.X = list(map(midi2numpy, tqdm(midifiles, desc='Loading MIDIs')))  # list of pieces
        self.n_features = self.X[0].shape[1]
        lengths = np.fromiter(map(len, self.X), dtype=int)  # their lengths (in 16th notes)
        lengths = lengths - seq_len + 1  # number of possible samples of length 'seq_len' for each piece
        self.piece_beginnings = np.insert(np.cumsum(lengths), 0, 0)  # first piece start at 0
        self.n_samples = self.piece_beginnings[-1]
        self.dataset = np.arange(self.n_samples)
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(self.n_samples) / float(self.batch_size))

    def __getitem__(self, idx):
        samples = self.dataset[idx * self.batch_size:(idx + 1) * self.batch_size]
        pieces_idx = np.searchsorted(self.piece_beginnings[1:], samples, side='right')  # find the piece the sample belongs to
        beginnings = samples - self.piece_beginnings[pieces_idx]  # find relative sample index
        endings = beginnings + self.seq_len

        # batch = np.empty(self.batch_size, self.seq_len, self.n_features)
        batch = [self.X[piece][begin:end, :] for piece, begin, end in zip(pieces_idx, beginnings, endings)]
        batch = np.stack(batch)

        x = batch[:, :-1, :]
        y = batch[:, 1:, :]
        return x, y

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.dataset)  # shuffle the dataset


if __name__ == '__main__':
    midifiles = get_midifiles('midi/Joplin/')[1:2]
    print(midifiles)
    dataset = MidiDataset(midifiles)
    x, y = dataset[0]
    from parse_ragtime import numpy2midi
    numpy2midi(x[0], 'x.mid')
    numpy2midi(y[0], 'y.mid')

