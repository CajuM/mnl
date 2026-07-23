#!/usr/bin/env python

import mnl

import sys

import matplotlib
import torch
import torchaudio

from matplotlib import pyplot as plt

matplotlib.use('TkAgg')

def main(signal_f):
    signal, sample_rate = torchaudio.load(signal_f)
    signal = signal.mean(0)

    plt.plot(signal.cpu())
    plt.show()

    spectorgram = torchaudio.transforms.Spectrogram(n_fft=mnl.N_FFT, power=1).cpu()

    y = spectorgram(signal.repeat(1000)).cpu().mean(-1)
    x = torch.arange(y.shape[0]) / y.shape[0] * mnl.SAMPLE_RATE

    plt.plot(x, y)
    plt.show()

if __name__ == '__main__':
    main(sys.argv[1])

