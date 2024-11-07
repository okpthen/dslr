from load import load
import pandas as pd
from logreg_train import train_list, estimateHouse

def prepare_data():
    df = load("datasets/dataset_test.csv")
    # df = df.dropna(subset=df.columns[6:])
    # df = df.reset_index(drop=True)#欠損データ破棄
    # df = df.fillna(df.min())#最低点で埋める
    std_data = pd.DataFrame()
    for col_index in range(6, df.shape[1]):
        data = df.iloc[:,col_index]
        std = df.iloc[:,col_index].std()
        mean = data.mean()
        # std_data[df.columns[col_index]] = ((data - mean) / std).fillna(0)# 欠損を0
        std_data[df.columns[col_index]] = ((data - mean) / std)#欠損処理なし

    return std_data

def longreg_predict():
    i = 0
    df = prepare_data()
    weight_data = pd.read_csv("predict/wehgit_data.csv", index_col=0)
    estimate = {}
    for index in range(len(df)):
        for key in train_list.keys():
            estimate[key] = estimateHouse(df, weight_data, key, index)
        if max(estimate.values()) < 0.75:
            i += 1
        # print(max(estimate.values()))
        # print(estimate["Ravenclaw"], estimate["Slytherin"],estimate["Hufflepuff"],estimate["Gryffindor"],)
        estimate_series = pd.Series(estimate)
        max_key = estimate_series.idxmax()
        df.loc[index, "Hogwarts House"] = max_key
    print(f"under 75% = {i}/{len(df)}({i / len(df) * 100}%)")
    df.to_csv("predict/precict.csv")
    house_data = df["Hogwarts House"]
    house_data.to_csv("house.csv")

longreg_predict()