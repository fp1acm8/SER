''' The file contains the functions useful for loading audio sample datasets.
    The file can be implemented with functions similar to that defined for the EMOVO dataset 
    but customized according to the structure of the desired dataset.

The functions are:
    > EMOVO_metadata()
'''
import pandas as pd
import numpy as np

import os

def EMOVO_metadata():
    '''It returns the file path, the actor, the pronounced sentence and the emotion 
    for each audio sample as a DataFrame. 
    '''
    EMOVO='data/EMOVO/'
    emovo_directory_list = os.listdir(EMOVO)

    file_emotion = []
    file_path = []
    file_text = []
    file_actor = []

    for dir in emovo_directory_list:
        actor = os.listdir(EMOVO + dir)
        for file in actor:
            part = file.split('.')[0]
            part = part.split('-')

            file_emotion.append(part[0])
            file_text.append(part[2])
            file_path.append(EMOVO + dir + '/' + file)
            file_actor.append(dir)

    # dataframe for emotion of files
    emotion_df = pd.DataFrame(file_emotion, columns=['Emotion'])
    # dataframe for text of files
    text_df = pd.DataFrame(file_text, columns=['Sentence'])
    # dataframe for actor of files
    actor_df = pd.DataFrame(file_actor, columns=['Actor'])
    # dataframe for path of files
    path_df = pd.DataFrame(file_path, columns=['Path'])

    EMOVO_df = pd.concat([emotion_df, text_df, actor_df, path_df], axis=1)

    # changing abbreviations to actual emotions
    EMOVO_df.Emotion.replace({'neu':'neutral', 
                            'dis':'disgust', 
                            'gio':'joy', 
                            'pau':'fear', 
                            'rab':'anger', 
                            'sor':'surprise', 
                            'tri':'sadness' }, inplace=True)

    return EMOVO_df