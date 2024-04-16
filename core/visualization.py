import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import queue
import logging
from matplotlib.gridspec import GridSpec
from matplotlib.figure import Figure
from core.calculations import Iteration, CorrectDetectionGraph
from typing import Union


logger = logging.getLogger(__name__)
matplotlib.use("Agg")


def process_calculation_queue(
    q: queue.Queue[Union[Iteration, CorrectDetectionGraph, None]]
) -> None:
    i = 0

    while True:
        r = q.get()

        if r is None:
            return

        fig = plt.figure(constrained_layout=True, figsize=(15, 10))

        # if fig.canvas.manager:
        #     fig.canvas.manager.set_window_title("График правильных детекций")

        render_plots(fig, r)

        fig.savefig(f"./res/{i}.webp")


def render_plots(fig: Figure, r: Union[Iteration, CorrectDetectionGraph]) -> None:
    gs = GridSpec(3, 5, figure=fig)

    match r:
        case Iteration():
            # Histograms
            ax1 = fig.add_subplot(gs[0, 1])
            ax1.plot(r.histograms[0])
            ax1.set_title("Histogram: Test")

            ax2 = fig.add_subplot(gs[1, 1])
            ax2.plot(r.histograms[1])
            ax2.set_title("Histogram: Reference")

            # DFT
            ax3 = fig.add_subplot(gs[0, 2])
            ax3.plot(np.log(np.abs(r.dfts[0])))
            ax3.set_title("DFT: Test")

            ax4 = fig.add_subplot(gs[1, 2])
            ax4.plot(np.log(np.abs(r.dfts[1])))
            ax4.set_title("DFT: Reference")

            # DCT
            ax5 = fig.add_subplot(gs[0, 3])
            ax5.plot(r.dcts[0])
            ax5.set_title("DCT: Test")

            ax6 = fig.add_subplot(gs[1, 3])
            ax6.plot(r.dcts[1])
            ax6.set_title("DCT: Reference")

            # Gradient
            ax7 = fig.add_subplot(gs[0, 4])
            ax7.plot(r.gradients[0])
            ax7.set_title("Gradient: Test")

            ax8 = fig.add_subplot(gs[1, 4])
            ax8.plot(r.gradients[1])
            ax8.set_title("Gradient: Reference")

            # Images
            ax9 = fig.add_subplot(gs[0, 0])
            ax9.imshow(r.images[0], cmap="gray")
            ax9.axis("off")
            ax9.set_title("Image: Test")

            ax10 = fig.add_subplot(gs[1, 0])
            ax10.imshow(r.images[1], cmap="gray")
            ax10.axis("off")
            ax10.set_title("Image: Reference")

        case CorrectDetectionGraph():
            correct_detections_percentage = round(
                sum([r.ys[i] for i in range(len(r.xs))]) / len(r.xs) * 100, 2
            )

            ax = fig.add_subplot(gs[2, 2:3])
            ax.plot(r.xs, r.ys, label=f"{correct_detections_percentage}%")
            ax.set_title("Процент правильных детекций")
            ax.legend(loc="lower right")
