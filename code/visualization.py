import numpy as np
import matplotlib as plot

def scatter_plot(data_x, data_y, x_axis_min, x_axis_max, y_axis_min, y_axis_max):

    plot.axes([x_axis_min, x_axis_max, y_axis_min, y_axis_max])
    plot.scatter(data_x, data_y)
    plot.show

    return