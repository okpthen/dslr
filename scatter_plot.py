from load import load
import matplotlib.pyplot as plt

color_map = {
        'Gryffindor': 'red',
        'Hufflepuff': '#ffff00',
        'Ravenclaw': 'blue',
        'Slytherin': 'green',
    }


def make_scatter_plot(df, categories, index_1, index_2):
    for category, color in color_map.items():
        category_data = df[categories == category]
        plt.scatter(category_data.iloc[:, index_1], category_data.iloc[:, index_2],
                    color=color, label=category, alpha=0.6)
    name = "scatter_plot/" + df.columns[index_1] + "_" + df.columns[index_2]  + ".png"
    plt.xlabel(df.columns[index_1])
    plt.ylabel(df.columns[index_2])
    plt.legend(title='Categories')
    plt.title(df.columns[index_1] + " vs " + df.columns[index_2])
    plt.savefig(name)
    plt.close()


def scatter_plot():
    df = load("datasets/dataset_train.csv")
    categories = df.iloc[:, 1] 
    for col_index_1 in range(6, df.shape[1]):
        for col_index_2 in range(6, df.shape[1]):
            if col_index_1 == col_index_2:
                continue
            make_scatter_plot(df ,categories, col_index_1, col_index_2)

scatter_plot()