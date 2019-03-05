import numpy as np
import matplotlib.pyplot as pyplot

def plot_data(plot, x, y):
    plot.plot(x[y==0, 0], x[y==0, 1], 'ob', alpha=0.5)
    plot.plot(x[y==1, 0], x[y==1, 1], 'xr', alpha=0.5)
    plot.legend(['Cluster 0', 'Cluster 1'])
    return plot
