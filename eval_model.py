#!/usr/bin/env python

import mnl

import sys

import torch

def main(model_f):
    model = torch.load(model_f, weights_only=False).to(mnl.device)

    loss, frame_v, audibility, score = model.evaluate()
    print(f'loss={loss.item()} frame_v={frame_v} audibility={audibility.item()} score={score}')

if __name__ == '__main__':
    main(sys.argv[1])
