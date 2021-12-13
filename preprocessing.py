import pandas as pd
import numpy as np

import modules.load_dataset as ld
from modules.data_preparation import store_features, label_manager

# Loading EMOVO
EMOVO_df = ld.EMOVO_metadata()

# Feature Extraction with noise and stretch & pitch for each audio sample
features_df = store_features(EMOVO_df.Path, EMOVO_df.Emotion, noise=True, stretch_pitch=True)
features_df.to_csv('checkpoints/EMOVO_features.csv', index=False) # store .csv file to avoid runnig the script again

# Labels setup
#features_df = label_manager(features_df, delate=[], rename={})