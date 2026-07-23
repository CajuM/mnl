#!/usr/bin/env python

import mnl

import json
import sys

import torch

def main(n_symbols, samples_per_symbol, dec_layers_lens, steps):
    model, optimizer = mnl.get_model(n_symbols, samples_per_symbol, dec_layers_lens)

    loss = -1.
    step = 0

    model.train()
    while (step < steps) or (score < 0.):
        if (step % 1024) == 0:
            print(step)

        model, loss, frame_v, audibility, score = mnl.train(model, optimizer)

        step += 1

    torch.save(model, 'model')
    print(f'loss={loss.item()} frame_v={frame_v} audibility={audibility.item()} score={score}')

if __name__ == '__main__':
    main(**json.loads(sys.argv[1]))
