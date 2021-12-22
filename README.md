# Speech Emotion Recognition
> **Speech Emotion Recognition** (SER) can be defined as extraction of the emotional state of the speaker from his or her speech signal

The project aims to develop a Speech Emotion Recognition system starting from EMOVO Corpus audio samples (italian language).\
The model used is an MLP implemented in Keras.

The goal of this repository is to create a starting base that can be implemented with the use of other datasets and other deep architectures.

---
## Structure

```
├── data/                                    
│   └── EMOVO/                                // EMOVO dataset
├── docs/                                     // documentation
├── models/                                  
│   ├── mlp.py                                // Multilayer Perceptron 
│   └── trained_model.m                       // trained model
├── modules/                                 
│   ├── load_dataset.py                       // loading EMOVO
│   ├── data_augmengtation.py                 // data augmentation functions
│   └── data_preparation.py                   // feature extraction and data preparation for ML
├── out/
│   ├── classification_report.csv             // statistics of predictions vs. true values
│   ├── confusion_matrix.png                  // confusion matrix 
│   └── predictions.csv                       // raw predictions
├── preprocessing.py                          // loading and preprocessing the dataset 
├── train.py                                  // training the model
└── predict.py                                // making predictions
```
## Requirements
Install dependencies:
```pip install -r requirements.txt```

## The dataset
[EMOVO](https://github.com/fp1acm8/SER/blob/main/docs/EMOVO_Corpus.pdf) is the first emotional corpus applicable to the Italian language. It is a database built from the voices of 6 actors (3 males and 3 females) who played 14 sentences simulating 6 emotional states (disgust, fear, anger, joy, surprise, sadness) plus the neutral state.

## Project pipeline
1. [`preprocessing.py`](https://github.com/fp1acm8/SER/blob/main/preprocessing.py):\
Load the dataset and extract the audio features through the `librosa` library. Synthetic data were created in order to increase the number of audio samples through data augmentation techniques.
2. [`train.py`](https://github.com/fp1acm8/SER/blob/main/train.py):\
Train the MLP classifier defined in [`modules/mlp.py`](https://github.com/fp1acm8/SER/blob/main/models/mlp.py). But first the data must be prepared so that it can be input to the neural network. To do this, a special function called [`data_preparation()`](https://github.com/fp1acm8/SER/blob/main/modules/data_preparation.py) has been created.
3. [`predict.py`](https://github.com/fp1acm8/SER/blob/main/predict.py):\
Make predictions on data never seen by the model. Summary data of the predictions made can be found in the folder [`out/`](https://github.com/fp1acm8/SER/blob/main/out/).

---
## Notes
* > Despite the promising results, the work can be improved by increasing the number of audio samples to train SER models. Having more data could allow you to train DNN able to perform feature extraction automatically (i.e., CNN and LSTM).
* > A possible business application of SER system was proposed in my master thesis [*"A Speech Emotion Recognition system to perform Sentiment Analysis in a business context"*](https://github.com/fp1acm8/SER/blob/main/docs/SER_businesscase.pdf).
