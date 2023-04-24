#!/usr/bin/env python

import mnl

import sys

import matplotlib
import torch
import torchaudio

from matplotlib import pyplot as plt

matplotlib.use('TkAgg')

signal_f = sys.argv[1]

signal, sample_rate = torchaudio.load(signal_f)
signal = signal.mean(0)
signal = torchaudio.functional.resample(signal, sample_rate, mnl.SAMPLE_RATE)

plt.plot(signal.cpu())
plt.show()

plt.plot(torch.arange(1000) / 1000 * 48000 / 2, torchaudio.transforms.Spectrogram(n_fft=1999, power=1)(signal.repeat(1000)).mean(dim=-1).cpu())
plt.show()
