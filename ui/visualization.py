import matplotlib.pyplot as plt
import numpy as np
import queue
import tkinter as tk
import logging
from matplotlib.gridspec import GridSpec
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from core import Iteration, CorrectDetectionGraph
from typing import Union, Literal


logger = logging.getLogger(__name__)
queue_polling_interval_ms = 5


def render_plots(
    window: tk.Tk, q: queue.Queue[Union[Iteration, CorrectDetectionGraph, None]]
) -> None:
    fig = plt.figure(constrained_layout=True, figsize=(15, 10))

    if fig.canvas.manager:
        fig.canvas.manager.set_window_title("График правильных детекций")

    plots = setup_plots(fig)

    logger.info("opening new window with plots")

    plt.show(block=False)

    update_plots(window, fig, plots, q)


type PlotID = Literal[
    "histogram_test",
    "histogram_reference",
    "dft_test",
    "dft_reference",
    "dct_test",
    "dct_reference",
    "gradient_test",
    "gradient_reference",
    "image_test",
    "image_reference",
    "graph",
]

type Plots = dict[PlotID, Axes]


def setup_plots(fig: Figure) -> Plots:
    gs = GridSpec(3, 5, figure=fig)

    # Histograms
    ax1 = fig.add_subplot(gs[0, 1])
    ax1.plot(0)
    ax1.set_title("Histogram: Test")

    ax2 = fig.add_subplot(gs[1, 1])
    ax2.plot(0)
    ax2.set_title("Histogram: Reference")

    # DFT
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.plot(0)
    ax3.set_title("DFT: Test")

    ax4 = fig.add_subplot(gs[1, 2])
    ax4.plot(0)
    ax4.set_title("DFT: Reference")

    # DCT
    ax5 = fig.add_subplot(gs[0, 3])
    ax5.plot(0)
    ax5.set_title("DCT: Test")

    ax6 = fig.add_subplot(gs[1, 3])
    ax6.plot(0)
    ax6.set_title("DCT: Reference")

    # Gradient
    ax7 = fig.add_subplot(gs[0, 4])
    ax7.plot(0)
    ax7.set_title("Gradient: Test")

    ax8 = fig.add_subplot(gs[1, 4])
    ax8.plot(0)
    ax8.set_title("Gradient: Reference")

    # Images
    ax9 = fig.add_subplot(gs[0, 0])
    ax9.set_title("Image: Test")

    ax10 = fig.add_subplot(gs[1, 0])
    ax10.set_title("Image: Reference")

    # Graph
    ax11 = fig.add_subplot(gs[2, 2:3])
    ax11.plot(0)
    ax11.set_title("Correct detections")

    return {
        "histogram_test": ax1,
        "histogram_reference": ax2,
        "dft_test": ax3,
        "dft_reference": ax4,
        "dct_test": ax5,
        "dct_reference": ax6,
        "gradient_test": ax7,
        "gradient_reference": ax8,
        "image_test": ax9,
        "image_reference": ax10,
        "graph": ax11,
    }


def update_plots(
    window: tk.Tk,
    fig: Figure,
    plots: Plots,
    q: queue.Queue[Union[Iteration, CorrectDetectionGraph, None]],
) -> None:
    if not plt.fignum_exists(fig.number):  # type: ignore
        return

    try:
        r = q.get_nowait()

        if r is None:
            return

        match r:
            case Iteration():
                plots["histogram_test"].lines[0].set_data(
                    np.arange(len(r.histograms[0])), r.histograms[0]
                )
                plots["histogram_test"].relim()
                plots["histogram_test"].autoscale_view()

                plots["histogram_reference"].lines[0].set_data(
                    np.arange(len(r.histograms[1])), r.histograms[1]
                )
                plots["histogram_reference"].relim()
                plots["histogram_reference"].autoscale_view()

                transformed_test_dft = np.log(np.abs(r.dfts[0]))
                plots["dft_test"].lines[0].set_data(
                    np.arange(len(transformed_test_dft)), transformed_test_dft
                )
                plots["dft_test"].relim()
                plots["dft_test"].autoscale_view()

                transformed_reference_dft = np.log(np.abs(r.dfts[1]))
                plots["dft_reference"].lines[0].set_data(
                    np.arange(len(transformed_reference_dft)), transformed_reference_dft
                )
                plots["dft_reference"].relim()
                plots["dft_reference"].autoscale_view()

                plots["dct_test"].lines[0].set_data(
                    np.arange(len(r.dcts[0])), r.dcts[0]
                )
                plots["dct_test"].relim()
                plots["dct_test"].autoscale_view()

                plots["dct_reference"].lines[0].set_data(
                    np.arange(len(r.dcts[1])), r.dcts[1]
                )
                plots["dct_reference"].relim()
                plots["dct_reference"].autoscale_view()

                plots["gradient_test"].lines[0].set_data(
                    np.arange(len(r.gradients[0])), r.gradients[0]
                )
                plots["gradient_test"].relim()
                plots["gradient_test"].autoscale_view()

                plots["gradient_reference"].lines[0].set_data(
                    np.arange(len(r.gradients[1])), r.gradients[1]
                )
                plots["gradient_reference"].relim()
                plots["gradient_reference"].autoscale_view()

                plots["image_test"].imshow(r.images[0], cmap="gray")
                plots["image_reference"].imshow(r.images[1], cmap="gray")

            case CorrectDetectionGraph():
                correct_detections_percentage = round(
                    sum([r.ys[i] for i in range(len(r.xs))]) / len(r.xs) * 100, 2
                )

                plots["graph"].lines[0].set_label(f"{correct_detections_percentage}%")
                plots["graph"].lines[0].set_data(r.xs, r.ys)
                plots["graph"].legend(loc="lower right")
                plots["graph"].relim()
                plots["graph"].autoscale_view()

        fig.canvas.draw_idle()
        fig.canvas.flush_events()
    except queue.Empty:
        pass

    window.after(queue_polling_interval_ms, update_plots, window, fig, plots, q)


def render_average_chart(l_square_side: int) -> None:
    pass
    # average_y = []
    # average_x = range(1, 10)

    # for i in average_x:
    #     sum = 0

    #     database = db.build(l_square_side)

    #     xs, ys = calculate_coordinates(database, i)

    #     for j in range(len(xs)):
    #         sum = sum + ys[j]

    #     average_y.append(sum / len(xs))

    # plt.subplot(2, 1, 2)
    # plt.plot(average_x, average_y, label=f"l={l_square_side}")
    # plt.title(
    #     "Plot 2: Средний процент правильных детекций для разного кол-ва эталонов",
    #     fontsize=10,
    #     y=1.1,
    # )
    # plt.legend(loc="lower right")
    # plt.show()
