#!/usr/bin/env python

import mnl

import math
import sys

import optuna

def objective(trial):
    n_symbols = trial.suggest_categorical('n_symbols', [2, 4, 16])
    n_bits = int(math.log2(n_symbols))

    samples_per_symbol = trial.suggest_int('samples_per_symbol', 20, 2400, step=20)
    n_dec_layers = trial.suggest_int('n_dec_layers', 1, 4)

    dec_layers_lens = []
    for dec_layer in range(n_dec_layers - 1):
        dec_layers_lens.append(trial.suggest_int(f'dec_layers_lens.{dec_layer}', 20, 100, step=20))

    model, optimizer = mnl.get_model(n_symbols, samples_per_symbol, dec_layers_lens)

    model.train()
    for step in range(mnl.HYP_STEPS):
        model, loss, frame_v, audibility, score = mnl.train(model, optimizer)
        trial.report(score, step)

        if (step % 1024) == 0:
            print(step)

    return frame_v

def main(n_trials):
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials)

    print("Best Hyperparameters:", study.best_params)

if __name__ == '__main__':
    main(int(sys.argv[1]))
