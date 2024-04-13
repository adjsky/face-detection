import cv2
import numpy as np
import numpy.typing as npt


H = 112
W = 92


def sc_scale(input: cv2.typing.MatLike, l: int):
    m = (int)(H / l)
    n = (int)(W / l)

    resize_forscale = cv2.resize(input, (n, m), interpolation=cv2.INTER_LINEAR)
    vector_scale = resize_forscale.ravel()

    return vector_scale


def zig_zag(C: npt.NDArray[np.signedinteger]):
    line: list[np.signedinteger] = []
    y = 0
    k = y

    while y < W:
        while k >= 0:
            line.append(C[y - k, k])
            k -= 1
        y += 1
        k = y

    return line


def dft(img: cv2.typing.MatLike):
    dft = cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT)  # type: ignore
    dft_shift = np.fft.fftshift(dft)

    magnitude_spectrum = 20 * np.log(
        cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1])
    )

    return zig_zag(magnitude_spectrum)


def dct(img: cv2.typing.MatLike):
    dct = cv2.dct(np.float32(img))  # type: ignore

    return zig_zag(dct)


def histogram(img: cv2.typing.MatLike) -> cv2.typing.MatLike:
    return cv2.calcHist([np.float32(img)], [0], None, [256], [0, 256])  # type: ignore


def gradient(img: cv2.typing.MatLike) -> list[float]:
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=5)
    abs_sobelx64f = np.absolute(sobelx)
    sobelx_8u = np.uint8(abs_sobelx64f)
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=5)
    abs_sobely64f = np.absolute(sobely)
    sobely_8u = np.uint8(abs_sobely64f)

    gradient = cv2.addWeighted(sobelx_8u, 0.5, sobely_8u, 0.5, 0)  # type: ignore

    result: list[float] = []

    for i in range(len(gradient)):
        result.append(round(sum(gradient[i]) / len(gradient[i]), 1))

    return result
