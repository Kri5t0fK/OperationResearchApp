from pathlib import Path
from typing import Dict
import PySimpleGUI as sg
import gui_data
import numpy as np
import matplotlib.pyplot as plt
import model

# needed for graph in gui:
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


def redraw_graph():
    # clear axis
    ax_graph.cla()

    # TEST PLOT DATA
    #TODO: plot ObjectiveFunction history here
    graph_data: np.ndarray = model_obj.graph_data

    # make fig and plot
    ax_graph.plot(graph_data[0], '-b')
    ax_graph.plot(graph_data[1], '--g')
    ax_graph.grid()
    ax_graph.set_xlim([0, model_obj.iteration_limit])
    # Instead of plt.show
    figure_canvas_agg.draw()


def load_file(window: sg.Window, vals: Dict):
    path = vals["_gui.input.file"]
    if path:
        path: Path = Path(path)
        model_obj.load_data(path)
        window['_gui.gen_init_solution'].update(disabled=False)
        sg.popup("File loaded!", font="Helvetica 13")
    else:
        sg.popup("Enter path!")
	

def pass_params_to_model(window: sg.Window, vals: Dict):
    alpha = float(vals['_gui.input.alpha'])
    beta = float(vals['_gui.input.beta'])
    gamma = float(vals['_gui.input.gamma'])
    model_obj.params = np.array([alpha, beta, gamma])
    model_obj.neighborhood_size = int(vals['_gui.input.nbrhd_size'])

    model_obj.iteration_limit = int(vals['_gui.input.max_iter'])
    model_obj.tabu_age[0] = int(vals['_gui.input.tabu_size'])
    model_obj.cutoff = float(vals['_gui.input.min_cutoff'])
    model_obj.aspiration_coeff = float(vals['_gui.input.aspiration_coeff'])
    return


def generate_initial_solution(window: sg.Window, vals: Dict):
    #TODO: initial solution
    pass_params_to_model(window, vals)
    model_obj.recipe_count = int(vals['_gui.input.recipe_count'])
    model_obj.generate_initial()

    window['_gui.optimise'].update(disabled=False)
    
    window['_gui.output.init_solution'].update(model_obj.initial_X[0])
    window['_gui.output.init_recipies'].update(np.sort(model_obj.initial_X[1]))

    split_cost: np.ndarray = model_obj.calculate_cost_function(model_obj.initial_X[1], split_mode=True)
    window['_gui.output.init_expenses'].update(split_cost[0])
    window['_gui.output.init_loss'].update(split_cost[1])
    window['_gui.output.init_time'].update(split_cost[2])
    
    sg.popup("Initial solution generated!")
    
def run_optimisation(window: sg.Window, vals: Dict):
    #TODO: run optimisation
    pass_params_to_model(window, vals)
    nbrhd_type = vals['_gui.input.nbrhd_type']
    solution_selection = vals['_gui.input.solution_select']
    stop_iter = model_obj.tabu_search(model_obj.iteration_limit, nbrhd_type, solution_selection)

    window['_gui.output.best_solution'].update(model_obj.global_best_X[1])
    window['_gui.output.stop_iter'].update(stop_iter)
    window['_gui.output.best_iter'].update(model_obj.global_best_X[0])
    window['_gui.output.best_recipies'].update(np.sort(model_obj.global_best_X[2]))

    split_cost: np.ndarray = model_obj.calculate_cost_function(model_obj.global_best_X[2], split_mode=True)
    window['_gui.output.best_expenses'].update(split_cost[0])
    window['_gui.output.best_loss'].update(split_cost[1])
    window['_gui.output.best_time'].update(split_cost[2])
    redraw_graph()
    sg.popup("Solution optimised!")


def main(window: sg.Window):

    while True:
        event, values = window.read()
        if event == "_gui.exit" or event == sg.WIN_CLOSED:
            break
        elif event == "_gui.file_select":
            load_file(window, values)
        elif event == "_gui.gen_init_solution":
            generate_initial_solution(window, values)
        elif event == "_gui.optimise":
            run_optimisation(window, values)

    window.close()


if __name__ == "__main__":
    # Window creation
    window: sg.Window = sg.Window(title="Kitchen Manager GUI", layout=gui_data.layout, finalize=True)

    model_obj = model.Model()

    # Graph setting (init)
    fig_graph = plt.figure(1)
    ax_graph = fig_graph.add_subplot(111)
    figure_canvas_agg = FigureCanvasTkAgg(fig_graph, window['_gui.canvas'].TKCanvas)
    figure_canvas_agg.get_tk_widget().pack(side='top', fill='both', expand=1)
    main(window)
