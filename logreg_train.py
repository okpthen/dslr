from load import load
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

train_list = {
    "Ravenclaw": ["Charms", "Muggle Studies", "Ancient Runes"],
    "Slytherin": ["Charms", "Divination", "Potions"],
    "Hufflepuff": ["Herbology", "Astronomy", "Ancient Runes"],
    "Gryffindor": ["Flying", "Transfiguration", "History of Magic"]
}
LearningRate = 0.1
AccRate = 0.985
Max = 100

def estimateHouse(df, weight_data, house, index):
    z = weight_data.loc[house, "slice"]
    subjects = train_list[house]
    for subject in subjects: 
        if pd.isna(df.loc[index, subject]):#欠損してたら0
            return 0
        z += df.loc[index, subject] * weight_data.loc[house, subject]
    return 1 / (1 + np.exp(-z))


def new_weight(df, weight_data, house):
    end = False
    tmpSum = {}
    m = len(df)
    tmpSum["slice"] = sum(estimateHouse(df, weight_data, house, i) - df.loc[i, house] for i in range(m))
    for subject in train_list[house]:
        tmpSum[subject] = sum((estimateHouse(df, weight_data, house, i) - df.loc[i, house]) * df.loc[i, subject] for i in range(m))
    
    for key in tmpSum.keys():
        weight_data.loc[house, key] -= LearningRate * tmpSum[key] / m

    acc = sum((1 if (1 if estimateHouse(df, weight_data, house, i) >= 0.75 else 0) == df.loc[i, house] else 0) for i in range(m))
    if acc / m > AccRate:
        end = True
    print(acc / m)
    return end, weight_data
    

def train(df, weight_data, key):
    number = 1
    while True:
        end, weight_data = new_weight(df, weight_data, key)
        if end == True:
            break
        if number == Max:
            break
        number += 1
    print(key, number)
    return weight_data


def prepare_data():
    df = load("datasets/dataset_train.csv")
    df = df.dropna()
    std_data = pd.DataFrame()
    data = df.iloc[:,1]
    std_data["House"] = df.iloc[:,1]
    for key in train_list.keys():
        std_data[key] = data.apply(lambda house: 1 if house == key else 0)
    for col_index in range(6, df.shape[1]):
        data = df.iloc[:,col_index]
        std = df.iloc[:,col_index].std()
        mean = data.mean()
        std_data[df.columns[col_index]]  = (data - mean) / std

    std_data = std_data.reset_index(drop=True)
    # std_data.to_csv("describe/dataset_prepare.csv")
    return std_data


def longer_train():
    df = prepare_data()
    # print(len(df))
    # print(df)
    weight_data = pd.DataFrame(index=["Ravenclaw", "Slytherin", "Gryffindor", "Hufflepuff"])
    for key in train_list.keys():
        weight_data.loc[key, "slice"] = 0.00
        for value in train_list[key]:
            weight_data.loc[key, value] = 0.00

    for key in train_list.keys():
        # for value in train_list[key]:
        weight_data = train(df, weight_data, key)

    # weight_data = train(df, weight_data, "Gryffindor")

    # print(weight_data)
    weight_data.to_csv("predict/wehgit_data.csv")
    print(weight_data)

if __name__ == "__main__":
    longer_train()