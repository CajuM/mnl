#!/usr/bin/env python

import mnl

import sys

import torch

device = torch.device('cuda')
model = mnl.get_model().to(device)

model_f = sys.argv[1]

model.load_state_dict(torch.load(model_f))

loss, success, audible = mnl.evaluate(model, mnl.get_dataloader(mnl.EVAL_BATCHES))
print(model_f, loss, success, audible)
