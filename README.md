# Speech Emotion Recognition

Developing a Speech Emotion Recognition system starting from EMOVO Corpus audio samples (italian language).
The model used is an MLP implemented in Keras.

## Structure

```
├── data/                                    
│   ├── EMOVO/                                // EMOVO dataset
├── docs/                                    
├── models/                                  
│   ├── mlp.py                                // MLP 
│   ├── trained_model.m                       // trained model
├── modules/                                 
│   ├── load_dataset.py                       // loading EMOVO
│   ├── data_augmengtation.py                 // data augmentation functions
│   ├── data_preparation.py                   // feature extraction and data preparation for ML
├── out/
│   ├── classification_report.csv             // statistics of predictions vs. true values
│   ├── confusion_matrix.png                  // confusion matrix 
│   ├── predictions.csv                       // raw predictions
├── preprocessing.py                          // loading and preprocessing the dataset 
├── train.py                                  // training the model
├── predict.py                                // making predictions
```