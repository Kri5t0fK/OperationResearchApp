import PySimpleGUI as sg
from model import NeighborhoodType, SolutionSelectionMethod

# Frame with parameters - inputs
frame_parameters = [
    [sg.Column([
        [sg.Text("alpha (Expenses):", size=(14, 1)), sg.Input("1", key="_gui.input.alpha", size=(6, 1))],
        [sg.Text("beta (Loss):", size=(14, 1)), sg.Input("1", key="_gui.input.beta", size=(6, 1))],
        [sg.Text("gamma (Time):", size=(14, 1)), sg.Input("1", key="_gui.input.gamma", size=(6, 1))],
        [sg.Text("Neighborhood:")],
        [sg.Text("- Type (Hamming):", size=(14, 1)), sg.Spin([i for i in NeighborhoodType], key="_gui.input.nbrhd_type", initial_value=NeighborhoodType(2), size=(8, 1))],
        [sg.Text("- Size :", size=(14, 1)), sg.Input("10", key="_gui.input.nbrhd_size", size=(6, 1))]
    ]),
    sg.VSeparator(),
    sg.Column([
        [sg.Text("Recipe count:", size=(13, 1)), sg.Input("3", key="_gui.input.recipe_count", size=(6, 1))],
        [sg.Text("Max iterations:", size=(13, 1)), sg.Input("100", key="_gui.input.max_iter", size=(6, 1))],
        [sg.Text("Tabu size:", size=(13, 1)), sg.Input("10", key="_gui.input.tabu_size", size=(6, 1))],
        [sg.Text("Solution selection:", size=(13, 1)), sg.Spin([i for i in SolutionSelectionMethod], key="_gui.input.solution_select", initial_value=SolutionSelectionMethod(1), size=(8, 1))],
        [sg.Text("Min Cutoff:", size=(13, 1)), sg.Input("0", key="_gui.input.min_cutoff", size=(6, 1))],
        [sg.Text("Aspiration coeff:", size=(13, 1)), sg.Input("0.9", key="_gui.input.aspiration_coeff", size=(6, 1))]
    ])]
]

# Frame with initial solution - outputs
frame_initial_output = [
    [sg.Column([
        [sg.Text("Initial solution:", size=(10, 1)), sg.Input("##", key="_gui.output.init_solution", size=(12, 1), disabled=True)],
    ])],
    [sg.Text("Initial Recipies:", size=(12, 1)), sg.Input("#, #, #, #, ...", key="_gui.output.init_recipies", disabled=True)],
    [
        sg.Text("Expenses:"), sg.Input("##", key="_gui.output.init_expenses", size=(10, 1), disabled=True),
        sg.Text(" "*2 + "Loss:"), sg.Input("##", key="_gui.output.init_loss", size=(10, 1), disabled=True),
        sg.Text(" "*2 + "Time:"), sg.Input("##", key="_gui.output.init_time", size=(10, 1), disabled=True)
    ]
]

# Frame with optimised solution - outputs
frame_optimised_output = [
    [sg.Column([
        [sg.Text("Best solution:", size=(10, 1)), sg.Input("##", key="_gui.output.best_solution", size=(12, 1), disabled=True)]
    ]),
    sg.VSeparator(),
    sg.Column([
        [sg.Text("Stop iteration:", size=(16, 1)), sg.Input("##", key="_gui.output.stop_iter", size=(6, 1), disabled=True)],
        [sg.Text("Best solution iteration:", size=(16, 1)), sg.Input("##", key="_gui.output.best_iter", size=(6, 1), disabled=True)]
    ])],
    [sg.Text("Optimal Recipies:", size=(12, 1)), sg.Input("#, #, #, #, ...", key="_gui.output.best_recipies", disabled=True)],
    [
        sg.Text("Expenses:"), sg.Input("##", key="_gui.output.best_expenses", size=(10, 1), disabled=True),
        sg.Text(" "*2 + "Loss:"), sg.Input("##", key="_gui.output.best_loss", size=(10, 1), disabled=True),
        sg.Text(" "*2 + "Time:"), sg.Input("##", key="_gui.output.best_time", size=(10, 1), disabled=True)
    ]
]

# Column with inputs & outputs
col_in_out = [
    [sg.Text("Input & Output")],
    [sg.Text("Load file with Kitchen Data (Recipes, Ingredients, Prices, etc.):")],
    [sg.Input("", key="_gui.input.file", size=30), sg.FileBrowse("Select file", target="_gui.input.file"), sg.Button("Load selected file", key="_gui.file_select")],
    [sg.Button("Generate Initial Solution", key="_gui.gen_init_solution", font="Helvetica 15", disabled=True), sg.Button("Optimise Solution", key="_gui.optimise", font="Helvatica 15", disabled=True)],
    [sg.Frame(title='Optimisation Parameters', layout=frame_parameters, relief=sg.RELIEF_SUNKEN, tooltip='Use these to set flags')],
    [sg.Text("")],
    [sg.Frame(title='Initial Solution - Output', layout=frame_initial_output, relief=sg.RELIEF_SUNKEN, tooltip='Use these to set flags')],
    [sg.Frame(title='Optimised Solution - Output', layout=frame_optimised_output, relief=sg.RELIEF_SUNKEN, tooltip='Use these to set flags')],
    [sg.Button("Exit", key="_gui.exit"), sg.Button("DEBUG Redraw!", key="_gui.draw")]
]

# Column with graph
col_graph = [
    [sg.Text("Graph")],
    [sg.Canvas(key="_gui.canvas")]
]

# Full layout
layout = [
    [sg.Column(col_in_out), sg.VSeparator(), sg.Column(col_graph)]
]
