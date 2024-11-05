import pandas as pd


def load(path: str):
    df = pd.read_csv(path)
    return df