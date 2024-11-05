from load import load
import pandas as pd


def statistics(series, statics, name):
    count = series.count()
    mean = series.mean()
    std = series.std()
    min_val = series.min()
    q25 = series.quantile(0.25)
    median = series.median()
    q75 = series.quantile(0.75)
    max_val = series.max()
    statics[name] = [count, mean, std, min_val, q25, median, q75, max_val]


def describe():
    df = load("datasets/dataset_train.csv")
    statics = pd.DataFrame(index=["Count","Mean","Std","Min","25%","50%","75%","Max"])
    for col_index in range(6, df.shape[1]):
        statistics(df.iloc[:, col_index], statics,  df.columns[col_index])
    statics.to_csv("describe/statics.csv")
    print(statics)

describe()