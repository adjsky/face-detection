import tkinter as tk
import threading
import logging
from visualization import render_plots, render_average_chart
from core import calculate_correct_detections
from queue import Queue


logging.basicConfig(
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    # stream=sys.stdout,
    filename="ui.log",
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


def build_chart(window: tk.Tk, l_square_side: int, n_stands: int) -> None:
    logger.info("building chart")

    q = Queue()

    threading.Thread(
        target=lambda: calculate_correct_detections(l_square_side, n_stands, q),
    ).start()

    render_plots(window, q)


if __name__ == "__main__":
    window = tk.Tk()
    window.title("Программа для моделирования систем распознавания людей по лицам")
    window.geometry("800x450")

    n_stands_label = tk.Label(window, text="Количество эталонов:")
    n_stands_label.pack()

    n_stands_scale = tk.Scale(window, from_=1, to=9, orient=tk.HORIZONTAL)
    n_stands_scale.pack()

    l_square_side_label = tk.Label(window, text="Сторона квадрата l:")
    l_square_side_label.pack()

    l_square_side_scale = tk.Scale(window, from_=1, to=10, orient=tk.HORIZONTAL)
    l_square_side_scale.pack()

    build_chart_button = tk.Button(
        window,
        text="Построить график правильных детекций",
        command=lambda: build_chart(
            window, int(l_square_side_scale.get()), int(n_stands_scale.get())
        ),
    )
    build_chart_button.pack()

    build_average_chart_button = tk.Button(
        window,
        text="Построить график зависимости процента правильных детекций от количества эталонов",
        command=lambda: render_average_chart(
            int(l_square_side_scale.get()),
        ),
    )
    build_average_chart_button.pack()

    logger.info("entering the main loop")

    window.mainloop()
