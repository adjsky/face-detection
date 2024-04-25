import logging
import click
import matplotlib
import matplotlib.pyplot as plt
import os
import shutil
import cv2
import numpy as np
from core.calculations import calculate_correct_detections
from matplotlib.gridspec import GridSpec
from matplotlib.figure import Figure
from core.calculations import Iteration

matplotlib.use("Agg")

logging.basicConfig(
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    filename="cli.log",
    level=logging.DEBUG,
)

modules_to_ignore_logs = ["matplotlib.font_manager", "PIL.PngImagePlugin", "PIL.Image"]

for module in modules_to_ignore_logs:
    logger = logging.getLogger(module)
    logger.disabled = True

logger = logging.getLogger(__name__)
result_folder = "res"
iterations_folder = "iterations"
correct_detections_graph_filename = "graph"


@click.group()
def cli() -> None:
    pass


@cli.command()
@click.option("--n-stands", "-n", required=True, help="количество эталонов", type=int)
@click.option(
    "--l-square-side", "-l", required=True, help="сторона квадрата l", type=int
)
def build_chart(n_stands: int, l_square_side: int) -> None:
    logger.info("building chart")

    (iterations, cdg) = calculate_correct_detections(l_square_side, n_stands)

    correct_detections_percentage = round(
        sum([cdg.ys[i] for i in range(len(cdg.xs))]) / len(cdg.xs) * 100, 2
    )

    click.echo(f"Процент правильных детекций: {correct_detections_percentage}%")

    if os.path.exists(result_folder):
        shutil.rmtree(result_folder)

    os.makedirs(os.path.join(result_folder, iterations_folder))

    click.echo(f"Используем папку `{result_folder}` для сохранения результатов.")
    click.echo(
        f"Сохраняем график правильных детекций в файл `{correct_detections_graph_filename}.webp`."
    )

    graph_fig, ax = plt.subplots()
    ax.plot(cdg.xs, cdg.ys, label=f"{correct_detections_percentage}%")
    ax.set_title("Процент правильных детекций")
    ax.legend(loc="lower right")

    graph_fig.savefig(
        f"{os.path.join(result_folder, correct_detections_graph_filename)}.webp"
    )

    click.echo(f"Сохраняем результаты итераций в папку `{iterations_folder}`.")

    iterations_fig = plt.figure(constrained_layout=True, figsize=(15, 10))

    with click.progressbar(iterations) as bar:
        for i in bar:
            process_iteration(iterations_fig, i)


def render_plots(fig: Figure, r: Iteration) -> None:
    gs = GridSpec(2, 6, figure=fig)

    # Histograms
    ax1 = fig.add_subplot(gs[0, 1])
    ax1.plot(r.histograms[0])
    ax1.set_title("Histogram: Test")

    ax2 = fig.add_subplot(gs[1, 1])
    ax2.plot(r.histograms[1])
    ax2.set_title("Histogram: Reference")

    # DFT
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.imshow(r.dfts[0][1], cmap="gray")
    ax3.set_title("DFT: Test")

    ax4 = fig.add_subplot(gs[1, 2])
    ax4.imshow(r.dfts[1][1], cmap="gray")
    ax4.set_title("DFT: Reference")

    # DCT
    ax5 = fig.add_subplot(gs[0, 3])
    ax5.imshow(
        cv2.normalize(np.log(abs(r.dcts[0][1]) + 1), None, 0, 1, cv2.NORM_MINMAX),  # type: ignore
        cmap="gray",
    )
    ax5.set_title("DCT: Test")

    ax6 = fig.add_subplot(gs[1, 3])
    ax6.imshow(
        cv2.normalize(np.log(abs(r.dcts[1][1]) + 1), None, 0, 1, cv2.NORM_MINMAX),  # type: ignore
        cmap="gray",
    )
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

    # Images
    ax11 = fig.add_subplot(gs[0, 5])
    ax11.imshow(r.sc_scales[0][1], cmap="gray")
    ax11.set_title("Scale: Test")

    ax12 = fig.add_subplot(gs[1, 5])
    ax12.imshow(r.sc_scales[1][1], cmap="gray")
    ax12.set_title("Scale: Reference")


def process_iteration(fig: Figure, i: Iteration) -> None:
    render_plots(fig, i)
    fig.savefig(f"{os.path.join(result_folder, iterations_folder, i.key)}.webp")
    fig.clear()


if __name__ == "__main__":
    cli()
