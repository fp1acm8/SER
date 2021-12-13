''' DATA AUGMENTATION
    Data augmentation in data analysis are techniques used to increase the amount of data by adding 
    slightly modified copies of already existing data or newly created synthetic data from existing data.

    Techniques used:
    >   noise injection
    >   shifting time
    >   changing pitch
    >   changing speed
'''

import numpy as np

import librosa

def noise(data):
    noise_amp = 0.035*np.random.uniform()*np.amax(data)
    data = data + noise_amp*np.random.normal(size=data.shape[0])
    return data

def stretch(data, rate=0.8):
    return librosa.effects.time_stretch(data, rate)

def shift(data):
    shift_range = int(np.random.uniform(low=-5, high = 5)*1000)
    return np.roll(data, shift_range)

def pitch(data, sampling_rate, pitch_factor=0.7):
    return librosa.effects.pitch_shift(data, sampling_rate, pitch_factor)