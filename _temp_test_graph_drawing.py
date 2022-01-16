import PySimpleGUI as sg
import numpy as np
import matplotlib.pyplot as plt

# Note the matplot tk canvas import
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# VARS CONSTS:
window = False

AppFont = 'Any 16'
sg.theme('LightGrey')

layout = [[sg.Canvas(key='figCanvas')],
          [sg.Button('yes', font=AppFont),sg.Button('no', font=AppFont),sg.Button('Exit', font=AppFont)]]
window = sg.Window('Such Window',
                            layout,
                            finalize=True,
                            resizable=True,
                            element_justification="right")

# make fig and plot
fig = plt.figure(1)

figure_canvas_agg = FigureCanvasTkAgg(fig, window['figCanvas'].TKCanvas)
figure_canvas_agg.get_tk_widget().pack(side='top', fill='both', expand=1)
# def draw_figure(canvas, figure):
#     return figure_canvas_agg


def example():
    # plt.clf()
    dataSize = 1000
    xData = np.random.randint(100, size=dataSize)
    yData = np.linspace(0, dataSize, num=dataSize, dtype=int)
    plt.plot(xData, yData, '.k')
    # Instead of plt.show
    figure_canvas_agg.draw()


# MAIN LOOP
def main():
    while True:
        event, values = window.read()
        if event == sg.WIN_CLOSED or event == 'Exit':
            break
        if event == "yes":
            example()
        if event == "no":
            fig.clf()
            figure_canvas_agg.draw()
    window.close()

if __name__ == "__main__":
    main()
    