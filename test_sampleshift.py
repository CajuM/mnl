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
symbols = mnl.get_symbols(model)

symbols2 = torchaudio.functional.resample(symbols, mnl.SAMPLE_RATE, mnl.SAMPLE_RATE * mnl.SAMPLE_SHIFT)
for idx in range(mnl.SAMPLE_SHIFT):
    pad1 = torch.rand((symbols2.shape[0], mnl.HALF_SHIFT), device=device)
    pad2 = torch.rand((symbols2.shape[0], mnl.HALF_SHIFT), device=device)
    symbols3 = torch.cat((pad1, symbols2, pad2), dim=-1)
    symbols3 = symbols3[:, idx: idx + symbols2.shape[1]]
    symbols3 = torchaudio.functional.resample(symbols3, mnl.SAMPLE_RATE * mnl.SAMPLE_SHIFT, mnl.SAMPLE_RATE)

    print(model.decoder(symbols3).softmax(-1).argmax(-1)[:10])
