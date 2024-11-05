from load import load
import seaborn as sns
import matplotlib.pyplot as plt

color_map = {
        'Gryffindor': 'red',
        'Hufflepuff': '#ffff00',
        'Ravenclaw': 'blue',
        'Slytherin': 'green',
    }


def pair_plot():
    df = load("datasets/dataset_train.csv")
    data = df.iloc[:, 6:]
    data['House'] = df.iloc[:, 1]
    sns.pairplot(data, hue='House', palette=color_map, plot_kws={'alpha': 0.5})
    plt.title("pair plot")
    plt.savefig("pair_plot.png")



pair_plot()