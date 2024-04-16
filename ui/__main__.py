import tkinter as tk
import threading
import logging
from visualization import render_plots
from core import calculate_correct_detections
from queue import Queue
from typing import Literal


logging.basicConfig(
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
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


def build_chart(
    window: tk.Tk,
    l_square_side: int,
    n_stands: int,
) -> None:
    toggle_widgets(window, "disabled")

    logger.info("building chart")

    q = Queue()

    threading.Thread(
        target=lambda: calculate_correct_detections(l_square_side, n_stands, q),
    ).start()

    (fig, _) = render_plots(q)

    def on_close(_) -> None:
        toggle_widgets(window, "normal")

    fig.canvas.mpl_connect("close_event", on_close)


def toggle_widgets(window: tk.Tk, state: Literal["disabled", "normal"]) -> None:
    for widget in window.winfo_children():
        if isinstance(
            widget, (tk.Button, tk.Entry, tk.Checkbutton, tk.Radiobutton, tk.Scale)
        ):
            widget.config(state=state)


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
        name="build_chart_button",
        text="Построить график правильных детекций",
        command=lambda: build_chart(
            window, int(l_square_side_scale.get()), int(n_stands_scale.get())
        ),
    )
    build_chart_button.pack(pady=10)

    logger.info("entering the main loop")

    window.mainloop()
