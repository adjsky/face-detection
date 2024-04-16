import threading
import logging
import click
from core.visualization import process_calculation_queue
from core.calculations import calculate_correct_detections
from queue import Queue


logging.basicConfig(
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    filename="cli.log",
    level=logging.DEBUG,
)

modules_to_ignore_logs = [
    "matplotlib.font_manager",
    "PIL.PngImagePlugin",
]

for module in modules_to_ignore_logs:
    logger = logging.getLogger(module)
    logger.disabled = True

logger = logging.getLogger(__name__)


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

    q = Queue()

    t1 = threading.Thread(
        target=lambda: calculate_correct_detections(l_square_side, n_stands, q),
    )

    t2 = threading.Thread(target=lambda: process_calculation_queue(q))

    t1.start()
    t2.start()

    t1.join()
    t2.join()


if __name__ == "__main__":
    cli()
