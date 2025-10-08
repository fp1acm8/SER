
''' This file contains the function useful for processing audio samples in such a way as to have them 
in the right format for applying ML algorithms.

The functions are:
    > extract_features()
    > get_features()
    > store_features()
    > data_preparation()
    > label_manager()
'''
import numpy as np
import pandas as pd

import librosa
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.utils import shuffle
from joblib import dump, load

import modules.data_augmentation as da


def extract_features(y,sr):
    '''Given the audio file y and the sampling rate sr, the function extract the following features:
        >   chromagram
        >   root mean square error
        >   spectrel centroid
        >   spectral flatness
        >   spectral bandwidth
        >   spectral roll-off
        >   zero crossing rate
        >   mel-frequency cepstral coefficients (MFCCs)
    '''
    result = np.array([])

    chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr) # multiple dimensions (12,t)
    rmse = librosa.feature.rms(y=y)
    spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
    flatness = librosa.feature.spectral_flatness(y=y)
    spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    zcr = librosa.feature.zero_crossing_rate(y)
    mfcc = librosa.feature.mfcc(y=y, sr=sr) # multiple dimensions (20,t)

    result = np.hstack((np.mean(chroma_stft,axis=1), np.mean(rmse), np.mean(spec_cent),
                        np.mean(flatness), np.mean(spec_bw), np.mean(rolloff),
                        np.mean(zcr), np.mean(mfcc, axis=1)))

    return result


def get_features(path, noise: bool, stretch_pitch: bool, shift: bool):
    ''' The function loads the audio file and its sampling rate and then it applies extract_features() function.
    The function can also extract features from synthetic audio obtained by applying data augmentation techniques 
    to the original audio file (noise, stretch&pitch, shift).
    '''
    #load audio file and its sampling rate
    y, sr = librosa.load(path, sr=48000)

    # feature extraction without augmentation
    res1 = extract_features(y,sr)
    result = np.atleast_2d(res1)

    # data with noise
    if (noise==True):
        noise_data = da.noise(y)
        res2 = extract_features(noise_data, sr)
        result = np.vstack((result, np.atleast_2d(res2))) # stacking vertically

    # data with stretching and pitching
    if (stretch_pitch==True):
        new_data = da.stretch(y)
        data_stretch_pitch = da.pitch(new_data, sr)
        res3 = extract_features(data_stretch_pitch, sr)
        result = np.vstack((result, np.atleast_2d(res3))) # stacking vertically

    # data with shift
    if (shift==True):
        shift_data = da.stretch(y)
        res4 = extract_features(shift_data, sr)
        result = np.vstack((result, np.atleast_2d(res4))) # stacking vertically

    return result


def store_features(audio_path: pd.core.frame.DataFrame, emotion: pd.core.frame.DataFrame,
                   noise: bool=False, stretch_pitch: bool=False, shift: bool=False):
    '''The function extracts features and store them together with the corresponding label as DataFrame.
    The funtion requires as input audio file paths, the corresponding emotion and data augmentation techniques desired.
    '''

    X, Y = [], []
    for path, emotion in zip(audio_path, emotion):
        # extracting features
        features = get_features(path, noise, stretch_pitch, shift)
        # storing the results and the corresponding emotion
        for sample_features in features:
            X.append(sample_features)
            Y.append(emotion)

    # Store results as DataFrame
    features_df = pd.DataFrame(X)
    features_df['labels'] = Y

    return features_df

def data_preparation(data: pd.core.frame.DataFrame, Y_col: int=-1, 
                    standard_scaler: bool=True, one_hot_encoding: bool=False,
                    train_test: bool=True, split_rate: float=0.8,
                    cv: bool=False, n_fold: int=5,
                    shuffle: bool=True):

                    '''This function prepares the data in such a way that it is ready to train an ML algorithm.

                    Parameters:
                    > data: data as DataFrame
                    > Y_col: column index number of labels (default is -1)
                    > StandardScaler: if True (default) data has to be standardized through the StandardScaler
                    > OneHotEncoder: if True (default is False) labels have to be one-hot-encoded
                    > train_test: if True (default) the dataset has to be split in train and test set
                    > test_size: it determines the size of the train test with respect to the whole dataset (default is 0.8)
                    > cv: if True (default is False) K-Folds cross-validator object is created
                    > n_fold: number of splitting iterations in the cross-validator
                    > shuffle: Whether to shuffle the data during train_test_split and cross_validation (default is True)
                    '''
                    # Initialize variables
                    Y = data.iloc[:,Y_col].values
                    X = data.iloc[: ,:-1].values
                    kf = None
                    encoder = None

                    if standard_scaler==True:
                        scaler = StandardScaler()
                        X = scaler.fit_transform(X)

                    if one_hot_encoding==True:
                        encoder = OneHotEncoder()
                        Y = encoder.fit_transform(np.array(Y).reshape(-1,1)).toarray()
                        dump(encoder, 'checkpoints/encoder.joblib') 

                    if train_test==True:
                        x_train, x_test, y_train, y_test = train_test_split(X, Y, random_state=0, 
                                                                            shuffle = shuffle, 
                                                                            train_size = split_rate)
                        if cv==True:
                            kf = KFold(n_splits=n_fold, shuffle=shuffle, random_state=None)
                        
                        pd.DataFrame(x_train).to_csv('checkpoints/x_train.csv', index=False)
                        pd.DataFrame(y_train).to_csv('checkpoints/y_train.csv', index=False)
                        pd.DataFrame(x_test).to_csv('checkpoints/x_test.csv', index=False)
                        pd.DataFrame(y_test).to_csv('checkpoints/y_test.csv', index=False)
                        return x_train, x_test, y_train, y_test, kf
                    else:
                        if cv==True:
                            kf = KFold(n_splits=n_fold, shuffle=shuffle, random_state=None)

                        X.to_csv('out/X.csv', index=False)
                        Y.to_csv('out/Y.csv', index=False)
                        return X, Y, kf


def label_manager(data: pd.core.frame.DataFrame, Y_col: int=-1,
                 delate: list=[], rename: dict={}):
    '''Managing labels of a DataFrame.

    Parameters:
    > data: data as DataFrame
    > Y_col: column index number of labels (default is -1)
    > delate: list of labels to be removed
    > rename: dictionary of labels to be replaced with new names
    '''

    # Select labels
    Y = data.iloc[:,Y_col]

    # Delate rows with unwanted labels
    if len(delate)!=0 :
        for label_del in delate:
            data = data[Y!=label_del]

    # Rename labels according to the dictionary passed as parameter
    if len(rename)!=0:
        data.labels.replace(rename, inplace=True)

    return data