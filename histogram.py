from load import load
import matplotlib.pyplot as plt


color_map = {
        'Gryffindor': 'red',
        'Hufflepuff': '#ffff00',
        'Ravenclaw': 'blue',
        'Slytherin': 'green',
    }

def make_hiotogram(data, categories, title):
    for category in color_map.keys():
        plt.hist(data[categories == category], 
                 bins=30, 
                 color=color_map[category], 
                 alpha=0.6, 
                 label=category) 
    plt.xlabel('Score')
    plt.ylabel('Number of people')
    plt.legend(title='Categories') 
    plt.title(title)
    file_name = "histogram/" + title + ".png"
    plt.savefig(file_name)
    plt.close()
    

def histogram():
    df = load("datasets/dataset_train.csv")
    categories = df.iloc[:, 1] 
    for col_index in range(6, df.shape[1]):
        data = df.iloc[:,col_index]
        # std = df.iloc[:,col_index].std()
        # mean = data.mean() 
        # std_data = (data - mean) / std
        # make_hiotogram(std_data, categories, df.columns[col_index] + "_std")
        make_hiotogram(data, categories, df.columns[col_index])


histogram()