#!/usr/bin/env python

import mnl

import sys

import torch
import torchaudio

def main(model_f, signal_f, seq):
    model = torch.load(model_f, weights_only=False).cpu()

    signal, sample_rate = torchaudio.load(signal_f)
    signal = signal.mean(0)

    seq = eval(sys.argv[3])
    seq = torch.tensor(seq).cpu()

    n_samples = seq.shape[0] * model.samples_per_symbol

    for idx in range(signal.shape[0] - n_samples + 1):
        guess = signal[idx: idx + n_samples]
        guess = guess.reshape((-1, model.samples_per_symbol))
        guess = model.decoder(guess).argmax(-1)

        if (guess == seq).all(dim=-1):
            print(idx)

if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2], eval(sys.argv[3]))
