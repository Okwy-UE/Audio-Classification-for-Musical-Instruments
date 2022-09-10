import os
from tqdm import tqdm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from python_speech_features import mfcc, logfbank
import librosa

def plot_signal(signals):
    fig, axes = plt.subplots(2, 5, sharex = False, sharey = True, figsize=(20,5))
    fig.suptitle('Time Series', size=16)
    i = 0
    for x in range(2):
        for y in range(5):
            axes[x, y].set_title(list(signals.keys())[i])
            axes[x, y].plot(list(signals.values())[i])
            axes[x, y].get_xaxis().set_visible(False)
            axes[x, y].get_yaxis().set_visible(False)
            i += 1

def plot_fft(fft):
    fig, axes = plt.subplots(2, 5, sharex = False, sharey = True, figsize=(20,5))
    fig.suptitle('Fourier Transforms', size=16)
    i = 0
    for x in range(2):
        for y in range(5):
            data = list(fft.values())[i]
            Y, freq = data[0], data[1]
            axes[x, y].set_title(list(fft.keys())[i])
            axes[x, y].plot(freq, Y)
            axes[x, y].get_xaxis().set_visible(False)
            axes[x, y].get_yaxis().set_visible(False)
            i += 1


def plot_fbank(fbank):
    fig, axes = plt.subplots(2, 5, sharex = False, sharey = True, figsize=(20,5))
    fig.suptitle('Filter Bank Coefficients', size=16)
    i = 0
    for x in range(2):
        for y in range(5):
            axes[x, y].set_title(list(fbank.keys())[i])
            axes[x, y].imshow(list(fbank.values())[i], cmap = 'hot', interpolation = 'nearest')
            axes[x, y].get_xaxis().set_visible(False)
            axes[x, y].get_yaxis().set_visible(False)
            i += 1

def plot_mfcss(mfcss):
    fig, axes = plt.subplots(2, 5, sharex = False, sharey = True, figsize=(20,5))
    fig.suptitle('Mel Frequency Capstrum Coefficients', size=16)
    i = 0
    for x in range(2):
        for y in range(5):
            axes[x, y].set_title(list(mfcss.keys())[i])
            axes[x, y].imshow(list(mfcss.values())[i], cmap = 'hot', interpolation = 'nearest')
            axes[x, y].get_xaxis().set_visible(False)
            axes[x, y].get_yaxis().set_visible(False)
            i += 1

def envelope(y, rate, threshold): # Magnitude plot of signal
    mask = []
    y = pd.Series(y).apply(np.abs)
    y_mean = y.rolling(window = int(rate/10), min_periods=1, center=True).mean()
    for mean in y_mean:
        if mean > threshold:
            mask.append(True)
        else:
            mask.append(False)
    return mask

def calc_fft(y, rate):
    n = len(y)
    freq = np.fft.rfftfreq(n, d = 1/rate)
    Y = abs(np.fft.rfft(y)/n)
    return Y, freq

instruments_dict = {}
instruments_dict['Instrument'] = []
instruments_dict['Sound'] = []

# Create a Dataframe containing the sound files and their classes
path = 'C:/Users/Okwy Uwadoka/Documents/Python Codes/30 ML Projects/Audio Classification for Musical Instruments/Audio-Classification wavfiles/'
for fol in os.listdir(path):
    inner_path = path + fol + '/'
    for file in os.listdir(inner_path):
        instruments_dict['Instrument'].append(fol)
        instruments_dict['Sound'].append(file)
df = pd.DataFrame(instruments_dict)

# Create a csv file called 'Instruments.csv'
# if file does not exist write header 
if not os.path.isfile('instruments.csv'):
   df.to_csv('instruments.csv', header='column_names')
else: # else it exists so append without mentioning the header
   df.to_csv('instruments.csv', mode='a', header=False)

df.set_index('Sound', inplace = True)

# Read the data from each sound file
for fol in os.listdir(path):
    inner_path = path + fol + '/'
    for file in os.listdir(inner_path):
        rate, signal = wavfile.read(inner_path+file)
        df.at[file, 'length'] = signal.shape[0]/rate

# Get the length distribution of each instrument class and create a plot
classes = list(np.unique(df.Instrument)) 
class_dist = df.groupby(['Instrument'])['length'].mean()
fig, ax = plt.subplots()
ax.set_title('Class Length Distribution')
ax.pie(class_dist, labels = class_dist.index, autopct = '%1.1f%%', startangle = 90)
ax.axis('equal')
# plt.show()
df.reset_index(inplace=True)

# Visualisation of the audio data
signals, fft, fbank, mfccs = {}, {}, {}, {} 
for c in classes:
    wavefile = df[df.Instrument == c].iloc[0, 0]
    signal, rate = librosa.load(path + c + '/' + wavefile, sr = 44100)
    mask = envelope(signal, rate, 0.0005)
    signal = signal[mask]
    signals[c] = signal
    fft[c] = calc_fft(signal, rate)

    bank = logfbank(signal[:rate], rate, nfilt = 26, nfft = 1103).T
    fbank[c] = bank

    mel = mfcc(signal[:rate], rate, numcep = 13, nfilt = 26, nfft = 1103).T
    mfccs[c] = mel

plot_signal(signals)
# plt.show()

plot_fft(fft)
# plt.show()

plot_fbank(fbank)
# plt.show()

plot_mfcss(mfccs)
# plt.show()

# Clean all the audio files
if len(os.listdir('C:/Users/Okwy Uwadoka/Documents/Python Codes/30 ML Projects/Audio Classification for Musical Instruments/clean')) == 0:
    for fol in os.listdir(path):
        inner_path = path + fol + '/'
        for file in os.listdir(inner_path):
            fpath = 'clean/' + file
            signal, rate = librosa.load(inner_path + file, sr=16000) # Downsampling
            mask = envelope(signal, rate, 0.0005)
            wavfile.write(filename = 'C:/Users/Okwy Uwadoka/Documents/Python Codes/30 ML Projects/Audio Classification for Musical Instruments/clean/' + file, rate = rate, data = signal[mask])

# Now we have all cleaned files.