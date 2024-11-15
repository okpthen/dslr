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
ClassPassRate = 0.75
Diff = 1

# train_list = {
#     "Ravenclaw": ["Charms", "Muggle Studies", "Ancient Runes","Flying","Transfiguration","History of Magic", "Herbology", "Astronomy", "Divination", "Potions"],
#     "Slytherin": ["Charms", "Muggle Studies", "Ancient Runes","Flying","Transfiguration","History of Magic", "Herbology", "Astronomy", "Divination", "Potions"],
#     "Hufflepuff": ["Charms", "Muggle Studies", "Ancient Runes","Flying","Transfiguration","History of Magic", "Herbology", "Astronomy", "Divination", "Potions"],
#     "Gryffindor": ["Charms", "Muggle Studies", "Ancient Runes","Flying","Transfiguration","History of Magic", "Herbology", "Astronomy", "Divination", "Potions"]
# }
# train_list = {
#     "Ravenclaw": ["Charms", "Potions", "Astronomy", "Herbology"],
#     "Slytherin": ["Charms", "Potions", "Astronomy", "Herbology"],
#     "Hufflepuff": ["Charms", "Potions", "Astronomy", "Herbology"],
#     "Gryffindor": ["Charms", "Potions", "Astronomy", "Herbology"]
# }

def estimateHouse(df, weight_data, house, index):
    z = weight_data.loc[house, "slice"]
    subjects = train_list[house]

    if df.loc[index, subjects].isna().any():
        return 0

    df_values = df.loc[index, subjects].to_numpy()
    weights = weight_data.loc[house, subjects].to_numpy()
    z += np.dot(df_values, weights)
    return 1 / (1 + np.exp(-z))


def new_weight(df, weight_data, house):
    m = len(df)
    subjects = train_list[house]
    
    # 全行に対する予測値を一括で計算
    estimates = np.array([estimateHouse(df, weight_data, house, i) for i in range(m)])
    actuals = df[house].to_numpy()
    
    # sliceの更新
    tmpSum_bias = np.sum(estimates - actuals)

    # 各重みの更新（ベクトル化計算）
    df_values = df[subjects].to_numpy()  # 必要なデータをNumPy配列に変換
    tmpSum_weights = np.dot((estimates - actuals), df_values)

    # 更新（ベクトル化）
    weight_data.loc[house, "slice"] -= LearningRate * tmpSum_bias / m
    weight_data.loc[house, subjects] -= LearningRate * tmpSum_weights / m

    return weight_data
    

def train(df, weight_data, key):
    number = 1
    prev_loss = float('inf')

    while number <= Max:
        # 重み更新
        weight_data = new_weight(df, weight_data, key)

        # 損失関数（対数損失）を計算
        m = len(df)
        estimates = np.array([estimateHouse(df, weight_data, key, i) for i in range(m)])
        actuals = df[key].to_numpy()
        loss = -np.sum(actuals * np.log(estimates) + (1 - actuals) * np.log(1 - estimates)) / m

        # 収束判定
        if abs(prev_loss - loss) < Diff:
            break
        prev_loss = loss

        # 進捗出力
        print(f"Iteration {number}, Loss: {loss}")
        number += 1

    print(f"{key} training completed in {number} iterations")
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