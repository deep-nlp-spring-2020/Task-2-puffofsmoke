#  Copyright (c) 2020.
#  roman.grigorov@gmail.com aka PuffOfSmoke or dePuff
import torch


def read_infile(infile):
    words = []
    with open(infile, encoding='utf-8') as f:
        for line in f:
            columns = line.split("\t")
            # I wanna use first and second columns and don't worry about symbol case
            line_words = columns[:1]
            for word in line_words:
                if word.find(' ') == -1:
                    words.append(word)

    words = list(set(words))
    return words


def binary_accuracy(preds, y):
    rounded_preds = torch.round(torch.sigmoid(preds))
    correct = (rounded_preds == y).float()
    acc = correct.sum() / len(correct)
    return acc
