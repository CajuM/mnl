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

model_f = sys.argv[1]

model.load_state_dict(torch.load(model_f))
symbols = mnl.get_symbols(model).cpu()

torchaudio.save('test/0.wav', symbols[0].reshape((1, -1)), mnl.SAMPLE_RATE, encoding='PCM_S', bits_per_sample=16)
torchaudio.save('test/m.wav', symbols.flatten().reshape(1, -1), mnl.SAMPLE_RATE, encoding='PCM_S', bits_per_sample=16)
