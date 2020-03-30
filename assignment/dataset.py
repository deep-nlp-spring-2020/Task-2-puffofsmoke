#  Copyright (c) 2020.
#  roman.grigorov@gmail.com aka PuffOfSmoke or dePuff
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset as TorchDataset


class Dataset(TorchDataset):
    """Custom data.Dataset compatible with data.DataLoader."""

    def __init__(self, data, vocab, token_bow, token_eow):
        self.data = data
        self.vocab = vocab
        self.token_bow = token_bow
        self.token_eow = token_eow

    def __getitem__(self, index):
        """
        Returns one tensor pair (source and target). The source tensor corresponds to the input word,
        with "BEGIN" and "END" symbols attached. The target tensor should contain the answers
        for the language model that obtain these word as input.
        """
        string = self.data[index]

        tensor = torch.zeros(len(string) + 2, dtype=torch.long)
        for ix in range(len(string)):
            tensor[ix+1] = self.vocab[string[ix]]

        tensor[0] = self.vocab[self.token_bow]
        tensor[-1] = self.vocab[self.token_eow]

        source, target = tensor[:-1], tensor[1:]
        return source, target

    def __len__(self):
        return len(self.data)


class Padder:
    def __init__(self, pad_code):
        self.pad_code = pad_code

    def __call__(self, batch):
        sorted_batch = sorted(batch, key=lambda x: x[0].shape[0], reverse=True)

        sequences = [x[0] for x in sorted_batch]
        sequences_padded = pad_sequence(sequences, batch_first=True, padding_value=self.pad_code)

        labels = [x[1] for x in sorted_batch]
        labels_padded = pad_sequence(labels, batch_first=True, padding_value=self.pad_code)

        return sequences_padded, labels_padded
