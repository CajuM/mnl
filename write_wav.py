#!/usr/bin/env python

import mnl

import sys

import torch
import torchaudio

def main(model_f, out_f, seq):
    model = torch.load(model_f, weights_only=False).to(mnl.device)

    seq = torch.tensor(eval(seq))
    seq = model.onehot(seq)
    seq = model.encoder(seq)
    seq = seq.flatten()

    torchaudio.save(out_f, seq, mnl.SAMPLE_RATE)

if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2], sys.argv[3])
