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
    file_name = "predict/" + title + ".png"
    plt.savefig(file_name)
    plt.close()
    

def histogram():
    df = load("predict/precict.csv")
    categories = df.iloc[:, 14]
    for col_index in range(1, 14):
        data = df.iloc[:,col_index]
        make_hiotogram(data, categories, df.columns[col_index])


histogram()