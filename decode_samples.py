#!/usr/bin/env python

import mnl

import sys

import matplotlib
import torch
import torchaudio

from matplotlib import pyplot as plt

matplotlib.use('TkAgg')

device = torch.device('cuda')

model = mnl.get_model().to(device)

signal_f = sys.argv[1]
signal, sample_rate = torchaudio.load(signal_f)
signal = signal.mean(0)
signal = torchaudio.functional.resample(signal, sample_rate, mnl.SAMPLE_RATE).to(device)

model_f = sys.argv[2]
seq = torch.tensor(eval(sys.argv[3])).to(device)

model.load_state_dict(torch.load(model_f))

for idx in range(mnl.SAMPLES_PER_SYMBOL):
    n_symbols = int((signal.shape[0] - idx) / mnl.SAMPLES_PER_SYMBOL)
    n_samples = n_symbols * mnl.SAMPLES_PER_SYMBOL

    signal2 = signal[idx: idx + n_samples].reshape((-1, mnl.SAMPLES_PER_SYMBOL))
    if not signal2.shape[0]:
        continue

    guess = model.decoder(signal2).argmax(-1)

    if (guess.shape[0] >= seq.shape[0]) and (guess.unfold(0, seq.shape[0], 1) == seq).all(dim=-1).any():
        plt.plot(guess.cpu())
        plt.show()
