import pandas as pd
import numpy as np

import modules 

EMOVO_df = modules.loading_EMOVO()
EMOVO_df.to_csv(r'fp1acm8/SER/data,
                index=False)