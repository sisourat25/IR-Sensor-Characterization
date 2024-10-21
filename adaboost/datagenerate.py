import numpy as np
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
import ast

def dataset():
    # Load the data from a CSV file
    data = pd.read_csv('../init-data.csv')
    
    # Features: IR sensor readings
    X = data[['left3', 'left2', 'left1', 'M', 'right1', 'right2', 'right3']].values
    
    # Labels: Zones (assuming they are lists like "['a1', 'b2']")
    # Convert string representations of lists to actual lists
    data['zones'] = data['zones'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
    
    # Initialize MultiLabelBinarizer
    mlb = MultiLabelBinarizer()
    y = mlb.fit_transform(data['zones'])
    
    return X, y, mlb