import cv2
import matplotlib.pyplot as plt
import numpy as np


def gradient_of_image(image):
    image = image.astype(np.float32)
    ksize = 5
    gX = cv2.Sobel(image, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=ksize)
    gY = cv2.Sobel(image, ddepth=cv2.CV_32F, dx=0, dy=1, ksize=ksize)
    # tries to kind of normalise the gradient, since brighter areas have higher gradient
    offset = 1
    return np.stack([gY / (image + offset), gX / (image + offset)], axis=2)


def filter_image(image: np.ndarray, scale_factor=4) -> np.ndarray:
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # blur image
    blur_size = (2 * int(0.007 * min(image.shape)) + 1,) * 2
    image = cv2.GaussianBlur(image, blur_size, 0)

    output_shape = (np.array(image.shape) * scale_factor).astype(np.int16)
    out_image = np.zeros(tuple(output_shape)).astype(np.uint8)

    gradient = gradient_of_image(image).astype(np.float32)
    gradient_magnitude = (gradient[:, :, 0] ** 2 + gradient[:, :, 1] ** 2) ** 0.5

    # scale gradient in range (0, 255)
    gradient_scaling_coefficient = np.quantile(gradient_magnitude, 0.999) / 200.0
    gradient_magnitude /= gradient_scaling_coefficient
    gradient_magnitude = np.clip(gradient_magnitude, 0, 255)

    gradient /= gradient_scaling_coefficient

    edge_coords = np.moveaxis(np.mgrid[0 : image.shape[0], 0 : image.shape[1]], 0, -1)[
        gradient_magnitude > np.quantile(gradient_magnitude, 0.8)
    ]
    N = min(len(edge_coords), 10000)
    edge_coords = edge_coords[np.random.randint(0, len(edge_coords), N)]

    line_length = 0.03 * np.min(image.shape)
    line_thickness = min(1, output_shape[0] // 500)
    for p in edge_coords:
        n = gradient[tuple(p)].astype(np.float32)
        n_len = gradient_magnitude[tuple(p)]
        if n_len < 0.0001:
            continue
        n_perpendicular = np.array([-n[1], n[0]]) / n_len

        color = int(((n_len / 255) ** 0.5) * 255)
        # color = n_len // 2
        start = np.flip(
            scale_factor * (p - (n_perpendicular * line_length)), axis=0
        ).astype(np.int16)

        end = np.flip(
            scale_factor * (p + (n_perpendicular * line_length)), axis=0
        ).astype(np.int16)

        cv2.line(
            out_image,
            start,
            end,
            color,
            line_thickness,
        )

    return out_image
