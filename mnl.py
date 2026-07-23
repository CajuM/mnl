#!/usr/bin/env python

import math
import os
import sys
import time

import optuna
import torch
import torchaudio

#TODO: Test on hardware

ARGS = {
    'SAMPLE_SHIFT': (int, 20),
    'SAMPLE_RATE': (int, 48000),
    'FREQUENCY': (int, 20000),
    'N_FFT': (int, 1999),
    'NOISE': (str, 'data/noise.wav'),
    'MIN_SNR': (float, 15.),
    'MAX_SNR': (float, 40.),
    'MIN_GAIN': (float, -20.),
    'MAX_GAIN': (float, 20.),
    'MAX_AUDIBILITY': (float, 0.01),
    'HYP_STEPS': (int, 10240),
    'TRAIN_BATCH': (int, 16),
    'FRAME_SIZE': (int, 64),
    'CHANNEL_NOISE': (bool, True),
    'CHANNEL_ATTENUATION': (bool, False),
    'CHANNEL_SAMPLE_SHIFT': (bool, True),
    'SIGNAL_AUDIBILITY': (bool, True),
}

DEC_LAYERS_LENS = [100, 100, 100]

for k, v in ARGS.items():
    if k in os.environ:
        v = ARGS[k][0](os.environ[k])

    else:
        v = ARGS[k][0](ARGS[k][1])

    globals()[k] = v
    print(f'{k}={v}')

device = torch.device('cpu')
torch.set_num_threads(14)

class OneHot(torch.nn.Module):
    def __init__(self, n_symbols):
        super().__init__()
        self.n_symbols = n_symbols

    def forward(self, x):
        x = torch.nn.functional.one_hot(x, num_classes=self.n_symbols)
        x = x.float()
        return x

class Encoder(torch.nn.Module):
    def __init__(self, n_symbols, samples_per_symbol):
        super().__init__()
        self.linear = torch.nn.Linear(n_symbols, samples_per_symbol)

    def forward(self, x):
        x = self.linear(x)
        x = x.sigmoid() * 2. - 1.
        return x

#TODO: Noise and attenuation distribution
class Channel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        noise, sample_rate = torchaudio.load(NOISE)
        noise = noise.mean(0)
        self.noise = torchaudio.functional.resample(noise, sample_rate, SAMPLE_RATE).to(device)

    def _sample_shift(self, x):
        x_len = x.shape[-1]
        x = torchaudio.functional.resample(x, SAMPLE_RATE, SAMPLE_RATE * SAMPLE_SHIFT)
        half_shift = SAMPLE_SHIFT // 2
        pad1 = torch.rand((x.shape[0], half_shift), device=device)
        pad2 = torch.rand((x.shape[0], half_shift), device=device)
        x = torch.cat((pad1, x, pad2), -1)
        shift = torch.randint(SAMPLE_SHIFT, (x.shape[0],), device=device)
        shift = shift.expand(x_len * SAMPLE_SHIFT, -1).T.flatten()
        base = SAMPLE_SHIFT * (x_len + 1) * torch.arange(x.shape[0], device=device).expand(x_len * SAMPLE_SHIFT, -1).T.flatten()
        inc =  torch.arange(x_len * SAMPLE_SHIFT, device=device).repeat(x.shape[0])
        shift = base + inc + shift
        x = x.flatten()[shift].reshape((x.shape[0], x_len * SAMPLE_SHIFT))
        x = torchaudio.functional.resample(x, SAMPLE_RATE * SAMPLE_SHIFT, SAMPLE_RATE)
        return x

    def _attenuation(self, x):
        gain_db = MIN_GAIN + (MAX_GAIN - MIN_GAIN) * torch.rand((x.shape[0], 1), device=device)
        x = x * (10. ** (gain_db / 20.))
        return x

    def _noise(self, x):
        x_len = x.shape[-1]
        rand_idx = torch.randint(self.noise.size(0) - x_len, (x.shape[0],), device=device)
        rand_idx = rand_idx.expand(x_len, -1).T.flatten()
        rand_idx = torch.arange(x_len, device=device).repeat(x.shape[0]) + rand_idx
        noise = self.noise[rand_idx].reshape(x.shape)
        snr =  MIN_SNR + (MAX_SNR - MIN_SNR) * torch.rand((x.shape[0], 1), device=device)
        snr = 10. ** (snr / 20.)
        x = (x * snr + noise) / (1. + snr)
        return x

    def forward(self, x):
        if CHANNEL_SAMPLE_SHIFT:
            x = self._sample_shift(x)

        if CHANNEL_ATTENUATION:
            x = self._attenuation(x)

        if CHANNEL_NOISE:
            x = self._noise(x)

        return x
        
#TODO: Take into consideration reducing the number of decoder weights/datatype for integration into ICs
class Decoder(torch.nn.Module):
    def __init__(self, n_symbols, samples_per_symbol, dec_layers_lens):
        super().__init__()

        dec_layers_lens.append(n_symbols)

        if len(dec_layers_lens) < 1:
            raise Exception(f'need more than one layer, provided={len(dec_layers_lens)}')

        seq = [torch.nn.Linear(samples_per_symbol, dec_layers_lens[0])]

        for layer in range(1, len(dec_layers_lens)):
            seq += [torch.nn.ReLU()]
            seq += [torch.nn.Linear(dec_layers_lens[layer - 1], dec_layers_lens[layer])]

        self.seq = torch.nn.Sequential(*seq)

    def forward(self, x):
        x = self.seq(x)
        x = x.softmax(-1)

        return x

class Model(torch.nn.Module):
    def __init__(self, n_symbols, samples_per_symbol, dec_layers_lens):
        super().__init__()

        self.n_symbols = n_symbols
        self.samples_per_symbol = samples_per_symbol
        self.n_bits = int(math.log2(n_symbols))

        self.onehot = OneHot(n_symbols)
        self.encoder = Encoder(n_symbols, samples_per_symbol)
        self.channel = Channel()
        self.decoder = Decoder(n_symbols, samples_per_symbol, dec_layers_lens)
        self.loss = torch.nn.CrossEntropyLoss()
        self.spectrogram = torchaudio.transforms.Spectrogram(n_fft=N_FFT, power=1).to(device)

    def _bandblock_audibility(self, samples):
        samples = samples.repeat(1, N_FFT // samples.shape[0] + 1)
        spectrogram = self.spectrogram(samples)
        bandpass = (spectrogram.shape[-2] * FREQUENCY * 2) // SAMPLE_RATE
        low_energy = spectrogram[:bandpass,]
        #high_energy = spectrogram[bandpass + 1:,].sum()
        energy = low_energy #+ high_energy
        return energy.mean()

    def forward(self, labels):
        y = self.onehot(labels)

        x = self.encoder(y)
        x = self.channel(x)
        x = self.decoder(x)

        loss = self.loss(x, y)
        outputs = x.argmax(-1)

        if SIGNAL_AUDIBILITY:
            symbols = self.get_audibility_sample()
            audibility = self._bandblock_audibility(symbols)

        else:
            audibility = torch.tensor(0.)

        return outputs, loss, audibility

    def evaluate(self):
        labels = torch.arange(self.n_symbols, device=device).repeat(TRAIN_BATCH)

        outputs, loss, audibility = self(labels)
        success = (outputs == labels).float().mean()

        frame_p = success.item() ** (FRAME_SIZE * 8 / self.n_bits)
        frame_v = (SAMPLE_RATE / self.samples_per_symbol * self.n_bits / 8 / FRAME_SIZE)

        frame_v = frame_v * frame_p

        if SIGNAL_AUDIBILITY and (audibility >= MAX_AUDIBILITY):
            return loss, frame_v, audibility, -audibility.item()

        else:
            return loss, frame_v, audibility, frame_v

    def get_symbols(self):
        symbols = torch.arange(self.n_symbols, device=device)

        symbols = self.onehot(symbols)
        symbols = self.encoder(symbols)

        return symbols

    def get_audibility_sample(self):
        a = torch.arange(self.n_symbols, device=device)
        b = torch.arange(self.n_symbols, device=device)

        symbols = self.onehot(torch.cartesian_prod(a, b).flatten())
        symbols = self.encoder(symbols)
        symbols = symbols.flatten()

        return symbols

def get_model(n_symbols, samples_per_symbol, dec_layers_lens):
    model = Model(n_symbols, samples_per_symbol, dec_layers_lens).to(device)
    optimizer = torch.optim.AdamW(model.parameters())

    return model, optimizer

def train(model, optimizer):
    loss, success, audibility, frame_v = model.evaluate()
    loss = audibility if SIGNAL_AUDIBILITY and (audibility >= MAX_AUDIBILITY) else loss

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return model, loss, success, audibility, frame_v
