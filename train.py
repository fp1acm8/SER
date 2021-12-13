import pandas as pd
import numpy as np

from models.mlp import MLPClassifier
from modules.data_preparation import data_preparation, label_manager


def cross_validation(X, Y, kf):
    # Initialize variables
    acc_per_fold =[]
    loss_per_fold =[]
    fold=1
    # Split the training data X and Y in n_folds as defined in the kf object
    for train_idx, val_idx in kf.split(X,Y):
        # Define the classifier
        classifier = MLPClassifier(n_class=Y.shape[1], input_shape=(X[train_idx].shape[1],))
        classifier.build()
        #Fit and evaluate training performances
        classifier.fit(X=X[train_idx], Y=Y[train_idx],
                        val_data=(X[val_idx], Y[val_idx]), 
                        callbacks=True)
        scores = classifier.eval(X=X[val_idx], Y=Y[val_idx])
        print(f'Score for fold {fold}:\
            {classifier.model.metrics_names[0]} of {scores[0]};\
            {classifier.model.metrics_names[1]} of {scores[1]*100}%')
        acc_per_fold.append(scores[1] * 100)
        loss_per_fold.append(scores[0])
        # Increase fold number
        fold = fold + 1
    # Compute mean values for accuracy and loss
    acc = np.mean(acc_per_fold)
    loss = np.mean(loss_per_fold)
    print('Accuracy of Model with ({}-Fold) Cross Validation is: {:.2f} %'.format(kf.n_splits,acc))
    print('Loss of Model with ({}-Fold) Cross Validation is: {:.4f}'.format(kf.n_splits,loss))
    # Save the model
    classifier.save(path='models',name='trained_model')


if __name__ == '__main__':
    # Load data
    data = pd.read_csv('checkpoints/EMOVO_features.csv')
    # Data preparation for ML
    x_train, x_test, y_train, y_test, kf= data_preparation( data, 
                                                            standard_scaler=True, one_hot_encoding=True,
                                                            train_test=True, split_rate=0.9, cv=True, n_fold=5)
    cross_validation(x_train, y_train, kf)










