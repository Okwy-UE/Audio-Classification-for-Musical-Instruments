import os
from scipy.io import wavfile
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, LSTM
from tensorflow.keras.layers import Dropout, Dense, TimeDistributed
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from sklearn.utils.class_weight import compute_class_weight
from tqdm import tqdm
from python_speech_features import mfcc
import pickle
from keras.callbacks import ModelCheckpoint
from cfg import Config

# Saving the data so it's not re-generated on each run
def check_data():
    if os.path.isfile(config.p_path):
        print(f'Loading existing data for {config.mode} model')
        with open(config.p_path, 'rb') as handle:
            tmp = pickle.load(handle)
            return tmp
    else:
        return None

def build_rand_feat():
    # Checking if there is some saved data for the model type (conv or time)
    tmp = check_data()
    if tmp:
        return tmp.data[0], tmp.data[1]
    # Generating new data
    X = []
    y = []
    _min, _max = float('inf'), -float('inf')
    for _ in tqdm(range(n_samples)):
        rand_class = np.random.choice(class_dist.index, p=prob_dist)
        file = np.random.choice(df[df.label == rand_class].index)
        rate, wav = wavfile.read('C:/Users/Okwy Uwadoka/Documents/Python Codes/30 ML Projects/Audio Classification for Musical Instruments/clean/' + file)
        label = df.at[file, 'label']
        rand_index = np.random.randint(0, wav.shape[0] - config.step)
        sample = wav[rand_index:rand_index + config.step]
        X_sample = mfcc(sample, rate, numcep = config.nfeat, nfilt=config.nfilt, nfft = config.nfft)
        _min = min(np.amin(X_sample), _min)
        _max = max(np.amax(X_sample, _max))
        X.append(X_sample)
        y.append(classes.index(label))
    config.min = _min
    config.max = _max
    X, y = np.array(X), np.array(y)
    X = (X - _min) / (_max - _min)  # Normalise the X array
    if config.mode == 'conv':
        X = X.reshape(X.shape[0], X.shape[1], X.shape[2], 1)
    elif config.mode == 'time':
        X = X.reshape(X.shape[0], X.shape[1], X.shape[2])
    y = to_categorical(y, num_classes = 10)
    config.data = (X, y)

    with open(config.p_path, 'wb') as handle:
        pickle.dump(config, handle, protocol = 2)
    return X, y

def get_conv_model():
    model = Sequential()
    model.add(Conv2D(16, (3,3), activation = 'relu', strides = (1,1), padding = 'same', input_shape = input_shape))
    model.add(Conv2D(32, (3,3), activation = 'relu', strides = (1,1), padding = 'same'))
    model.add(Conv2D(64, (3,3), activation = 'relu', strides = (1,1), padding = 'same'))
    model.add(Conv2D(128, (3,3), activation = 'relu', strides = (1,1), padding = 'same'))
    model.add(MaxPool2D((2,2)))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(128, activation = 'relu'))
    model.add(Dense(64, activation = 'relu'))
    model.add(Dense(10, activation = 'softmax'))
    model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['acc'])
    return model

def get_recurrent_model():
    # Shape of data for RNN is (n, time, feat) ==> (n by 9 by 13)
    model.add(LSTM(128, return_sequences = True, input_shape = input_shape))
    model.add(LSTM(128, return_sequences = True))
    model.add(Dropout(0.5))
    model.add(TimeDistributed(Dense(64, activation = 'relu')))
    model.add(TimeDistributed(Dense(32, activation = 'relu')))
    model.add(TimeDistributed(Dense(16, activation = 'relu')))
    model.add(TimeDistributed(Dense(8, activation = 'relu')))
    model.add(Flatten())
    model.add(Dense(10, activation = 'softmax'))
    model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['acc'])
    return model

# Load in the csv file
df = pd.read_csv('instruments.csv')
df.set_index('Sound', inplace = True)

# Load in the dataset
for f in df.index:
    rate, signal = wavfile.read('clean/' + f)
    df.at[f, 'length'] = signal.shape[0]/rate

# Get the classes and class distribution
classes = list(np.unique(df['Instrument']))
class_dist = df.groupby(['label'])['length'].mean()

# The number of samples we will use from our dataset
# will be twice the number of samples we will have if we split each audio file into
# small 0.1s chunks. 
n_samples = 2*int(df['length'].sum()/0.1)
prob_dist = class_dist / class_dist.sum()
choices = np.random.choice(class_dist.index, p = prob_dist)

# Plot the distribution
fig, ax = plt.subplots()
ax.set_title('Class Distribution')
ax.pie(class_dist, labels = class_dist.index, autopct = '%1.1f%%', startangle = 90)
ax.axis('equal')
# plt.show()

# Instantiating the config class to know what mode we are using
config = Config(mode = 'conv')

if config.mode == 'conv':
    X, y = build_rand_feat()
    y_flat = np.argmax(y, axis = 1)
    input_shape = (X.shape[1], X.shape[2], 1)
    model = get_conv_model()

elif config.mode == 'time':
    X, y = build_rand_feat()
    y_flat = np.argmax(y, axis = 1)
    input_shape = (X.shape[1], X.shape[2])
    model = get_recurrent_model()

# Computing class weights
class_weight = compute_class_weight('balanced', np.unique(y_flat), y_flat) # Helps to reduce bias

# Setting model checkpoints
checkpoint = ModelCheckpoint(config.model_path, monitor = 'val_acc', verbose = 1, mode = 'max', save_best_only = True, save_weights_only = False)

# Fitting and saving the model
model.fit(X, y, epochs = 20, batch_size = 32, shuffle=True, class_weight = class_weight, validaton_split = 0.1, callbacks = [checkpoint])
model.save(config.model_path)