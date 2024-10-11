import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

def dataset():
    
    data = pd.read_csv('../init-data.csv')
    
    # Features: IR sensor readings
    
    X = data[['left3', 'left2', 'left1', 'M', 'right1', 'right2', 'right3']].values
    
    y_raw = data['zones']
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_raw)
    
    return X, y, label_encoder