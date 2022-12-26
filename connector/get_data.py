import pandas as pd 
import numpy as np
from sklearn.datasets import make_classification

features = [f'FEATURE_{i}' for i in range(10)]

def get_dataset():
    return (
        make_classification(
            n_samples=45_000,
            n_features=10,
            n_classes=2
            )
    )

np.random.choice([""])



df = pd.DataFrame(columns=features)
df[features] = dataset[0]
df['TARGET'] = dataset[1]
print(df['TARGET'].value_counts())
