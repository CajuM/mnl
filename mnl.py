#!/usr/bin/env python

import math
import os
import sys
import time

import torch
import torchaudio

#TODO: Test on hardware
#TODO: Change the datatype to something supported by most microphones

ARGS = {
    'SAMPLE_SHIFT': (int, 20),
    'SAMPLE_RATE': (int, 48000),
    'HIGHPASS_HZ': (float, 19000.),
    'LOWPASS_HZ': (float, 20000.),
    'AUDIBLE_HZ': (float, 1000.),
    'SAMPLES_PER_SYMBOL': (int, 200),
    'N_FFT': (int, 1999),
    'N_SYMBOLS': (int, 2),
    'NOISE': (str, 'data/noise.wav'),
    'MIN_SNR': (float, -50.),
    'MAX_SNR': (float, 60.),
    'MIN_GAIN': (float, -60.),
    'MAX_GAIN': (float, 0.),
    'MAX_AUDIBILITY': (float, 0.001),
    'TRAIN_BATCHES': (int, 100),
    'EVAL_BATCHES': (int, 1000),
    'EVAL_ROUNDS': (int, 100),
    'USE_BANDPASS': (int, 0),
    'EPOCHS': (int, 1000000),
    'CHANNEL_NOISE': (int, 1),
    'CHANNEL_ATTENUATION': (int, 1),
    'CHANNEL_SAMPLE_SHIFT': (int, 1),
    'N_DATA': (int, 4),
    'N_PARITY': (int, 3),
}

for k, v in ARGS.items():
    if k in os.environ:
        v = ARGS[k][0](os.environ[k])

    else:
        v = ARGS[k][0](ARGS[k][1])

    globals()[k] = v
    print(f'{k}={v}')

device = torch.device('cuda')

#TODO: Filter audible signals
class OneHot(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = torch.nn.functional.one_hot(x, num_classes=N_SYMBOLS)
        x = x.float()
        return x

class Encoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(N_SYMBOLS, SAMPLES_PER_SYMBOL)

    def forward(self, x):
        x = self.linear(x)
        x = x.sigmoid() * 2. - 1.
        return x

class Parity(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(N_DATA * N_SYMBOLS, N_PARITY * SAMPLES_PER_SYMBOL)

    def forward(self, x):
        x = self.linear(x)
        x = x.sigmoid() * 2. - 1.
        return x

#TODO: Scale model "score" to noise and attenuation
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
        half_shift = int(SAMPLE_SHIFT / 2)
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
        x = x * (snr / (1. + snr)) + noise * (1. / (1. + snr))
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
    def __init__(self):
        super().__init__()
        self.seq = torch.nn.Sequential(
            torch.nn.Linear(SAMPLES_PER_SYMBOL * (N_DATA + N_PARITY), SAMPLES_PER_SYMBOL * (N_DATA + N_PARITY)),
            torch.nn.ReLU(),
            torch.nn.Linear(SAMPLES_PER_SYMBOL * (N_DATA + N_PARITY), SAMPLES_PER_SYMBOL * (N_DATA + N_PARITY)),
            torch.nn.ReLU(),
            torch.nn.Linear(SAMPLES_PER_SYMBOL * (N_DATA + N_PARITY), SAMPLES_PER_SYMBOL * (N_DATA + N_PARITY)),
            torch.nn.ReLU(),
            torch.nn.Linear(SAMPLES_PER_SYMBOL * (N_DATA + N_PARITY), N_SYMBOLS * N_DATA),
        )

    def forward(self, x):
        if USE_BANDPASS:
            x = torchaudio.functional.highpass_biquad(x, SAMPLE_RATE, HIGHPASS_HZ)
            x = torchaudio.functional.lowpass_biquad(x, SAMPLE_RATE, LOWPASS_HZ)

        x = self.seq(x)
        return x

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.onehot = OneHot()
        self.encoder = Encoder()
        self.parity = Parity()
        self.channel = Channel()
        self.decoder = Decoder()
        self.loss = torch.nn.CrossEntropyLoss()

        self.spectrogram = torchaudio.transforms.Spectrogram(n_fft=N_FFT, power=1).to(device)

        max_energy_signal = 2. * torch.sin(torch.arange(SAMPLES_PER_SYMBOL, device=device) * AUDIBLE_HZ * 2. * torch.pi / SAMPLE_RATE)
        max_energy_symbols = max_energy_signal.repeat(N_SYMBOLS, 1)
        self.max_audible_energy = self._energy(max_energy_symbols).item()

    def _energy(self, symbols):
        x = torch.arange(N_SYMBOLS, device=device)
        x = torch.cartesian_prod(x, x)
        x_rand = torch.randint(x.shape[0], (max(N_SYMBOLS * N_SYMBOLS, int((N_FFT + 1) / 2)),))
        x = x[x_rand,:].flatten()
        x = symbols[x, :].flatten()
        spectrogram = self.spectrogram(x)
        highpass = int(spectrogram.shape[-2] * HIGHPASS_HZ / SAMPLE_RATE * 2)
        lowpass = int(spectrogram.shape[-2] * LOWPASS_HZ / SAMPLE_RATE * 2)
        low_energy = spectrogram[:highpass,].sum()
        high_energy = spectrogram[lowpass:,].sum()
        energy = low_energy + high_energy
        return energy

    def forward(self, labels):
        labels = labels.reshape(-1)
        labels = self.onehot(labels)
        p = self.parity(labels.reshape((-1, N_SYMBOLS * N_DATA)))
        x = self.encoder(labels)
        x = x.reshape((-1, N_DATA * SAMPLES_PER_SYMBOL))
        x = torch.cat([x, p], -1)
        x = x.reshape((-1, SAMPLES_PER_SYMBOL))
        x = self.channel(x)
        x = x.reshape((-1, (N_DATA + N_PARITY) * SAMPLES_PER_SYMBOL))
        x = self.decoder(x)
        x = x.reshape((-1, N_SYMBOLS))

        loss = self.loss(x, labels)

        outputs = x.argmax(-1)
        outputs = outputs.reshape((-1, N_DATA))

        symbols = self.get_symbols()
        audible = self._energy(symbols) / self.max_audible_energy

        return outputs, loss, audible

    def get_symbols(self):
        symbols = self.onehot(torch.arange(N_SYMBOLS, device=device))
        symbols = self.encoder(symbols)
        return symbols

def get_model():
    return Model()

def get_dataloader(batches):
    data = torch.randint(N_SYMBOLS, (N_SYMBOLS * batches, N_DATA), device=device)

    while True:
        yield (data.clone(), data.clone())

def evaluate(model, dataloader):
    inputs, labels = next(dataloader)

    model.train(False)
    avg_success = 0
    for _ in range(EVAL_ROUNDS):
        outputs, loss, audible = model(inputs)
        success = (outputs == labels).float().mean()
        avg_success += success

    avg_success /= EVAL_ROUNDS
    model.train(True)

    return loss.item(), avg_success.item(), audible.item()

def main():
    model = get_model()
    model = model.to(device)
    model.train(True)

    trainloader = get_dataloader(TRAIN_BATCHES)
    evalloader = get_dataloader(EVAL_BATCHES)
    optimizer = torch.optim.AdamW(model.parameters())

    running_loss = 0.
    last_loss = 1e9
    last_time = time.time()
    idx = 1
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data

        outputs, loss, audible = model(inputs)
        loss = audible if audible >= MAX_AUDIBILITY else loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        if (time.time() - last_time) > 10.:
            last_time = time.time()

            with torch.no_grad():
                loss, success, audible = evaluate(model, evalloader)

            if loss < last_loss and audible < (MAX_AUDIBILITY * 1.1):
                last_loss = loss
                torch.save(model.state_dict(), f'models/model-{i + 1}.pt')
                print(f'[SAVED {i + 1}] running_loss={running_loss / idx}; loss={loss}; audible={audible}; success={success};')

            else:
                print(f'[{i + 1}] running_loss={running_loss / idx}; loss={loss}; audible={audible}; success={success};')

            running_loss = 0.
            idx = 1

        idx += 1

        if i > EPOCHS:
            break


    print('Finished Training')

if __name__ == '__main__':
    main()
