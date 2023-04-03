# Methods for Visualizing Data

import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt

def scatter_plot_all(data, data_labels, show_plot):

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    X = np.zeros(len(data))
    Y = np.zeros(len(data))
    Z = np.zeros(len(data))

    for i in range(len(data)):
        X[i] = data[i][0]
        Y[i] = data[i][1]
        Z[i] = data[i][2]

    ax.scatter(X, Y, Z, c=data_labels) 

    ax.set_xlabel('Component 0')
    ax.set_ylabel('Component 1')
    ax.set_zlabel('Component 2')

    plt.savefig('../results/figures/scatterplot.png')

    if show_plot:
        plt.show()

    return

def kde_map_all(data, data_labels, show_plot): 

    data_coc = np.concatenate((data, data_labels), axis = 1)
    dataframe = pd.DataFrame(data_coc, columns=["Component 0", "Component 1", "Component 2", "Label"])

    g = sb.FacetGrid(dataframe, col = "Label")
    g.map_dataframe(sb.kdeplot, x = "Component 0", color = 'r')
    g.map_dataframe(sb.kdeplot, x = "Component 1", color = 'g')
    g.map_dataframe(sb.kdeplot, x = "Component 2", color = 'b')
    g.set_xlabels("Component Values")
    
    
    plt.savefig('../results/figures/kde.png')

    if show_plot:
        plt.show()

    return

def pairplot_all(data, data_labels, show_plot):

    data_coc = np.concatenate((data, data_labels), axis = 1)
    dataframe = pd.DataFrame(data_coc, columns=["Component 0", "Component 1", "Component 2", "Label"])

    sb.pairplot(dataframe, hue = "Label")

    plt.savefig('../results/figures/pairplot.png')    
    
    if show_plot:
        plt.show()

    return

def heatmap(): ## ASK PROF

    return

