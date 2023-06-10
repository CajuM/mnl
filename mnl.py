#!/usr/bin/env python

import math
import sys
import time

import torch
import torchaudio

#TODO: Test on hardware
#TODO: Change the datatype to something supported by most microphones

HALF_SHIFT = 10
SAMPLE_SHIFT = 2 * HALF_SHIFT
SAMPLE_RATE = 48000
HIGHPASS_HZ = 19000.
LOWPASS_HZ = 20000.
AUDIBLE_HZ = 1000.
#TODO: How few?
SAMPLES_PER_SYMBOL = 350
N_FFT = 1999
N_SYMBOLS = 2
NOISE = 'data/noise.wav'
MIN_SNR = -50.
MAX_SNR = 60.
MIN_GAIN = -60.
MAX_GAIN = 0.
TRAIN_BATCHES = 100
EVAL_BATCHES = 1000
MAX_AUDIBILITY = .001
EVAL_ROUNDS = 100

device = torch.device('cuda')

#TODO: Filter audible signals
class Encoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(N_SYMBOLS, SAMPLES_PER_SYMBOL)

    def forward(self, x):
        x = torch.nn.functional.one_hot(x, num_classes=N_SYMBOLS)
        x = x.float()
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

    def forward(self, x):
        x = torchaudio.functional.resample(x, SAMPLE_RATE, SAMPLE_RATE * SAMPLE_SHIFT)
        pad1 = torch.rand((x.shape[0], HALF_SHIFT), device=device)
        pad2 = torch.rand((x.shape[0], HALF_SHIFT), device=device)
        x = torch.cat((pad1, x, pad2), -1)
        shift = torch.randint(SAMPLE_SHIFT, (x.shape[0],), device=device)
        shift = shift.expand(SAMPLES_PER_SYMBOL * SAMPLE_SHIFT, -1).T.flatten()
        base = SAMPLE_SHIFT * (SAMPLES_PER_SYMBOL + 1) * torch.arange(x.shape[0], device=device).expand(SAMPLES_PER_SYMBOL * SAMPLE_SHIFT, -1).T.flatten()
        inc =  torch.arange(SAMPLES_PER_SYMBOL * SAMPLE_SHIFT, device=device).repeat(x.shape[0])
        shift = base + inc + shift
        x = x.flatten()[shift].reshape((x.shape[0], SAMPLES_PER_SYMBOL * SAMPLE_SHIFT))
        x = torchaudio.functional.resample(x, SAMPLE_RATE * SAMPLE_SHIFT, SAMPLE_RATE)

        gain_db = MIN_GAIN + (MAX_GAIN - MIN_GAIN) * torch.rand((x.shape[0], 1), device=device)
        x = x * (10 ** (gain_db / 20))

        rand_idx = torch.randint(self.noise.size(0) - SAMPLES_PER_SYMBOL, (x.shape[0],), device=device)
        rand_idx = rand_idx.expand(SAMPLES_PER_SYMBOL, -1).T.flatten()
        rand_idx = torch.arange(SAMPLES_PER_SYMBOL, device=device).repeat(x.shape[0]) + rand_idx
        noise = self.noise[rand_idx].reshape(x.shape)
        snr =  MIN_SNR + (MAX_SNR - MIN_SNR) * torch.rand(x.shape[0], device=device)
        x = torchaudio.functional.add_noise(x, noise, snr)

        return x
        
#TODO: Take into consideration reducing the number of decoder weights/datatype for integration into ICs
class Decoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.seq = torch.nn.Sequential(
            torch.nn.Linear(SAMPLES_PER_SYMBOL, SAMPLES_PER_SYMBOL),
            torch.nn.ReLU(),
            torch.nn.Linear(SAMPLES_PER_SYMBOL, SAMPLES_PER_SYMBOL),
            torch.nn.ReLU(),
            torch.nn.Linear(SAMPLES_PER_SYMBOL, SAMPLES_PER_SYMBOL),
            torch.nn.ReLU(),
            torch.nn.Linear(SAMPLES_PER_SYMBOL, N_SYMBOLS),
        )

    def forward(self, x):
        x = self.seq(x)
        return x

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
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
        x = self.encoder(labels)
        x = self.channel(x)
        x = self.decoder(x)

        labels_oh = torch.nn.functional.one_hot(labels, num_classes=N_SYMBOLS).float()
        loss = self.loss(x, labels_oh)

        outputs = x.argmax(-1)

        symbols = get_symbols(self)
        audible = self._energy(symbols) / self.max_audible_energy

        return outputs, loss, audible

def get_model():
    return Model()

def get_dataloader(batches):
    data = torch.arange(N_SYMBOLS, device=device)
    data = data.repeat(batches)

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

def get_symbols(model):
    symbols = model.encoder(torch.arange(N_SYMBOLS, device=device))
    return symbols

def main():
    model = get_model()
    model = model.to(device)
    model.train(True)

    trainloader = get_dataloader(TRAIN_BATCHES)
    evalloader = get_dataloader(EVAL_BATCHES)
    optimizer = torch.optim.Adam(model.parameters())

    running_loss = 0.
    last_success = .9
    last_time = time.time()
    idx = 1
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data

        outputs, closs, audible = model(inputs)
        loss = audible if audible >= MAX_AUDIBILITY else closs

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        if (time.time() - last_time) > 10.:
            last_time = time.time()

            with torch.no_grad():
                closs, success, audible = evaluate(model, evalloader)

            print(f'[{i + 1}] running_loss={running_loss / idx}; closs={closs}; audible={audible}; success={success};')
            running_loss = 0.
            idx = 1

            if success > last_success and audible < (MAX_AUDIBILITY * 1.1):
                last_success = success
                torch.save(model.state_dict(), f'models/model-{i + 1}.pt')

        idx += 1

        if i > 1000000:
            break


    print('Finished Training')

if __name__ == '__main__':
    main()
