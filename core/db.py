import cv2
import os
import logging
import numpy as np
import numpy.typing as npt
from dataclasses import dataclass
from core.functions import dct, dft, histogram, gradient, sc_scale

references_folder = "references"
logger = logging.getLogger(__name__)

type Image = cv2.typing.MatLike
type ScScale = npt.NDArray
type DFT = list[np.signedinteger]
type DCT = list[np.signedinteger]
type Histogram = cv2.typing.MatLike
type Gradient = list[float]


@dataclass
class DB:
    images: list[list[Image]]
    sc_scales: list[list[ScScale]]
    dfts: list[list[DFT]]
    dcts: list[list[DCT]]
    histograms: list[list[Histogram]]
    gradients: list[list[Gradient]]


def build(l_square_side: int) -> DB:
    logger.info("building new database")

    images: list[list[Image]] = []
    sc_scales: list[list[ScScale]] = []
    dfts: list[list[DFT]] = []
    dcts: list[list[DCT]] = []
    histograms: list[list[Histogram]] = []
    gradients: list[list[Gradient]] = []

    with os.scandir(references_folder) as l1_entries:
        for l1_entry in l1_entries:
            if not l1_entry.is_dir():
                continue

            l1_images: list[Image] = []
            l1_sc_scales: list[ScScale] = []
            l1_dfts: list[DFT] = []
            l1_dcts: list[DCT] = []
            l1_histograms: list[Histogram] = []
            l1_gradients: list[Gradient] = []

            with os.scandir(f"{references_folder}/{l1_entry.name}") as l2_entries:
                for l2_entry in l2_entries:
                    if not l2_entry.is_file():
                        continue

                    stand_img = cv2.imread(
                        f"{references_folder}/{l1_entry.name}/{l2_entry.name}",
                        cv2.IMREAD_GRAYSCALE,
                    )

                    l1_images.append(stand_img)
                    l1_sc_scales.append(sc_scale(stand_img, l_square_side))
                    l1_dfts.append(dft(stand_img))
                    l1_dcts.append(dct(stand_img))
                    l1_histograms.append(histogram(stand_img))
                    l1_gradients.append(gradient(stand_img))

            images.append(l1_images)
            sc_scales.append(l1_sc_scales)
            dfts.append(l1_dfts)
            dcts.append(l1_dcts)
            histograms.append(l1_histograms)
            gradients.append(l1_gradients)

    return DB(images, sc_scales, dfts, dcts, histograms, gradients)
