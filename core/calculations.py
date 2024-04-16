import core.db as db
import cv2
import logging
from scipy.spatial.distance import cityblock
from dataclasses import dataclass
from queue import Queue
from typing import Union


logger = logging.getLogger(__name__)


@dataclass
class Iteration:
    images: tuple[db.Image, db.Image]
    sc_scales: tuple[db.ScScale, db.ScScale]
    dfts: tuple[db.DFT, db.DFT]
    dcts: tuple[db.DCT, db.DCT]
    histograms: tuple[db.Histogram, db.Histogram]
    gradients: tuple[db.Gradient, db.Gradient]


@dataclass
class CorrectDetectionGraph:
    xs: list[int]
    ys: list[float]


def calculate_correct_detections(
    l_square_side: int,
    n_stands: int,
    q: Queue[Union[Iteration, CorrectDetectionGraph, None]],
) -> None:
    logger.info("starting calculations")

    database = db.build(l_square_side)

    rights = 0
    alls = 0

    xs: list[int] = []
    ys: list[float] = []

    for i in range(len(database.images)):
        for j in range(n_stands, len(database.images[i])):
            min_dct = cityblock(database.dcts[i][j], database.dcts[0][0])

            class_dct = 0
            min_dft = cityblock(database.dfts[i][j], database.dfts[0][0])

            class_dft = 0
            min_scale = cityblock(
                database.sc_scales[i][j],
                database.sc_scales[0][0],
            )

            class_scale = 0
            min_histogram = cv2.compareHist(
                database.histograms[i][j],
                database.histograms[0][0],
                cv2.HISTCMP_BHATTACHARYYA,
            )

            class_histogram = 0
            min_gradient = cityblock(
                database.gradients[i][j],
                database.gradients[0][0],
            )
            class_gradient = 0

            for k in range(len(database.images)):
                for l in range(0, n_stands):
                    manhattan_dct = cityblock(
                        database.dcts[i][j],
                        database.dcts[k][l],
                    )

                    if manhattan_dct < min_dct:
                        min_dct = manhattan_dct
                        class_dct = k

                    manhattan_dft = cityblock(
                        database.dfts[i][j],
                        database.dfts[k][l],
                    )

                    if manhattan_dft < min_dft:
                        min_dft = manhattan_dft
                        class_dft = k

                    manhattan_scale = cityblock(
                        database.sc_scales[i][j],
                        database.sc_scales[k][l],
                    )

                    if manhattan_scale < min_scale:
                        min_scale = manhattan_scale
                        class_scale = k

                    manhattan_histogram = cv2.compareHist(
                        database.histograms[i][j],
                        database.histograms[k][l],
                        cv2.HISTCMP_BHATTACHARYYA,
                    )

                    if manhattan_histogram < min_histogram:
                        min_histogram = manhattan_histogram
                        class_histogram = k

                    manhattan_gradient = cityblock(
                        database.gradients[i][j],
                        database.gradients[k][l],
                    )

                    if manhattan_gradient < min_gradient:
                        min_gradient = manhattan_gradient
                        class_gradient = k

                    q.put(
                        Iteration(
                            (database.images[i][j], database.images[k][l]),
                            (database.sc_scales[i][j], database.sc_scales[k][l]),
                            (database.dfts[i][j], database.dfts[k][l]),
                            (database.dcts[i][j], database.dcts[k][l]),
                            (database.histograms[i][j], database.histograms[k][l]),
                            (database.gradients[i][j], database.gradients[k][l]),
                        )
                    )

            right_answer = 0

            if class_dct == i:
                right_answer += 1
            if class_dft == i:
                right_answer += 1
            if class_gradient == i:
                right_answer += 1
            if class_histogram == i:
                right_answer += 1
            if class_scale == i:
                right_answer += 1
            if right_answer > 2:
                rights += 1

            alls += 1

            xs.append(alls)
            ys.append(rights / alls)

            q.put(
                CorrectDetectionGraph(
                    xs[:],
                    ys[:],
                )
            )

    logger.info("finished calculations, sending the sentinel value")

    q.put(None)
