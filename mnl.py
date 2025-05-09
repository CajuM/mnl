#!/usr/bin/env python

import math
import os
import sys
import time

import torch
import torchaudio

#TODO: Test on hardware

ARGS = {
    'SAMPLE_SHIFT': (int, 20),
    'SAMPLE_RATE': (int, 48000),
    'FREQUENCY': (int, 20000),
    'AUDIBLE_HZ': (float, 1000.),
    'SAMPLES_PER_SYMBOL': (int, 2400),
    'N_SYMBOLS': (int, 2),
    'N_FFT': (int, 1999),
    'NOISE': (str, 'data/noise.wav'),
    'MIN_SNR': (float, -50.),
    'MAX_SNR': (float, 60.),
    'MIN_GAIN': (float, -60.),
    'MAX_GAIN': (float, 0.),
    'MAX_AUDIBILITY': (float, 71),
    'EVAL_ROUNDS': (int, 100),
    'EPOCHS': (int, 1000000),
    'CHANNEL_NOISE': (bool, True),
    'CHANNEL_ATTENUATION': (bool, True),
    'CHANNEL_SAMPLE_SHIFT': (bool, True),
    'SIGNAL_AUDIBILITY': (bool, True),
    'H_LEN1': (int, 100),
    'H_LEN2': (int, 100),
    'H_LEN3': (int, 100),
}

for k, v in ARGS.items():
    if k in os.environ:
        v = ARGS[k][0](os.environ[k])

    else:
        v = ARGS[k][0](ARGS[k][1])

    globals()[k] = v
    print(f'{k}={v}')

device = torch.device('cuda')

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
        self.linear = torch.nn.Linear(N_SYMBOLS, SAMPLES_PER_SYMBOL)

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
    def __init__(self, h_len1, h_len2, h_len3):
        super().__init__()
        self.seq = torch.nn.Sequential(
            torch.nn.Linear(SAMPLES_PER_SYMBOL, h_len1),
            torch.nn.ReLU(),
            torch.nn.Linear(h_len1, h_len2),
            torch.nn.ReLU(),
            torch.nn.Linear(h_len2, h_len3),
            torch.nn.ReLU(),
            torch.nn.Linear(h_len3, N_SYMBOLS),
        )

    def forward(self, x):
        x = self.seq(x)
        return x

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.onehot = OneHot()
        self.encoder = Encoder()
        self.channel = Channel()
        self.decoder = Decoder(H_LEN1, H_LEN2, H_LEN3)
        self.loss = torch.nn.CrossEntropyLoss()
        self.spectrogram = torchaudio.transforms.Spectrogram(n_fft=N_FFT, power=1).to(device)

    def _bandblock_audibility(self, samples):
        samples = samples.repeat(1, N_FFT // samples.shape[0] + 1)
        spectrogram = self.spectrogram(samples)
        bandpass = (spectrogram.shape[-2] * FREQUENCY * 2) // SAMPLE_RATE
        low_energy = spectrogram[:bandpass,].sum()
        high_energy = spectrogram[bandpass + 1:,].sum()
        energy = low_energy + high_energy
        return energy

    def forward(self, labels):
        y = self.onehot(labels)

        x = self.encoder(y)
        x = self.channel(x)
        x = self.decoder(x)

        loss = self.loss(x, y)
        outputs = x.argmax(-1)

        if SIGNAL_AUDIBILITY:
            symbols = self.get_symbols()
            audibility = self._bandblock_audibility(symbols)

        else:
            audibility = torch.tensor(0.)

        return outputs, loss, audibility

    def get_symbols(self):
        a = torch.arange(N_SYMBOLS, device=device)
        b = torch.arange(N_SYMBOLS, device=device)

        symbols = self.onehot(torch.cartesian_prod(a, b).flatten())
        symbols = self.encoder(symbols)
        symbols = symbols.flatten()

        return symbols

def get_model():
    return Model()

def get_dataloader(batches):
    while True:
        data = torch.randint(N_SYMBOLS, (batches,), device=device)
        yield data

def evaluate(model, dataloader):
    labels = next(dataloader)

    model.train(False)

    avg_success = 0.
    avg_audibility = 0.

    for _ in range(EVAL_ROUNDS):
        outputs, loss, audibility = model(labels)

        success = (outputs == labels).float().mean()
        avg_success += success

        avg_audibility += audibility

    avg_success /= EVAL_ROUNDS
    avg_audibility /= EVAL_ROUNDS

    model.train(True)

    return loss.item(), avg_success.item(), avg_audibility.item()

def main():
    model = get_model()
    model = model.to(device)
    model.train(True)

    trainloader = get_dataloader(16)
    evalloader = get_dataloader(128)
    optimizer = torch.optim.AdamW(model.parameters())

    running_loss = 0.
    last_loss = 1e9
    last_time = time.time()
    idx = 1
    for i, data in enumerate(trainloader, 0):
        labels = data

        _, loss, audiblity = model(labels)
        loss = audiblity if SIGNAL_AUDIBILITY and (audiblity >= MAX_AUDIBILITY) else loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        if (time.time() - last_time) > 10.:
            last_time = time.time()

            with torch.no_grad():
                loss, success, audibility = evaluate(model, evalloader)

            rep = f'running_loss={running_loss / idx}; loss={loss}; audibility={audibility}; success={success};'

            if (loss < last_loss) and (audibility <= MAX_AUDIBILITY):
                last_loss = loss
                torch.save(model.state_dict(), f'models/model-{i + 1}.pt')
                print(f'[SAVED {i + 1}] {rep}')

            else:
                print(f'[{i + 1}] {rep}')

            running_loss = 0.
            idx = 1

        idx += 1

        if i > EPOCHS:
            break

    print('Finished Training')

if __name__ == '__main__':
    main()
