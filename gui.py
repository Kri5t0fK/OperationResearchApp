import PySimpleGUI as sg
from numpy.core.fromnumeric import size
import gui_data
import numpy as np
import matplotlib.pyplot as plt
# needed for graph in gui:
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg




def redraw_graph():
    # clear axis
    ax_graph.cla()

    # TEST PLOT DATA
    #TODO: plot ObjectiveFunction history here
    dataSize = 1000
    xData = np.random.randint(100, size=dataSize)
    yData = np.linspace(0, dataSize, num=dataSize, dtype=int)

    # make fig and plot
    ax_graph.plot(xData, yData, '.k')
    # Instead of plt.show
    figure_canvas_agg.draw()


def load_file():
    #TODO: file loading
    sg.popup("File loaded!", font="Helvetica 13")

def generate_initial_solution():
    #TODO: initial solution
    sg.popup("Initial solution generated!")
    
def run_optimisation():
    #TODO: run optimisation
    sg.popup("Solution optimised!")


def main():

    while True:
        event, values = window.read()
        if event == "_gui.exit" or event == sg.WIN_CLOSED:
            break
        elif event == "_gui.file_select":
            load_file()
        elif event == "_gui.gen_init_solution":
            generate_initial_solution()
        elif event == "_gui.optimise":
            run_optimisation()
        elif event == "_gui.draw":
            redraw_graph()

    window.close()


if __name__ == "__main__":
    # Window creation
    window = sg.Window(title="Kitchen Manager GUI", layout=gui_data.layout, finalize=True)

    # Graph setting (init)
    fig_graph = plt.figure(1)
    ax_graph = fig_graph.add_subplot(111)
    figure_canvas_agg = FigureCanvasTkAgg(fig_graph, window['_gui.canvas'].TKCanvas)
    figure_canvas_agg.get_tk_widget().pack(side='top', fill='both', expand=1)
    main()
