import random
import sys

import cv2
import matplotlib.pyplot as plt
import numpy as np


def gradient_of_image(image):
    ksize = 5
    gX = cv2.Sobel(image, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=ksize)
    gY = cv2.Sobel(image, ddepth=cv2.CV_32F, dx=0, dy=1, ksize=ksize)
    return np.stack([gY, gX], axis=2)


def explore_edges(image):
    n = 5
    fig, ax = plt.subplots(n, n)
    for i in range(n):
        lower_bound = 10 + 4 * i
        for j in range(n):
            upper_bound = lower_bound + j * 4
            ax[i, j].imshow(compute_edges(image, lower_bound, upper_bound))
            ax[i, j].title.set_text(f"Lower {lower_bound}, upper {upper_bound}")
    plt.show()


def filter_image(image: np.ndarray, random_coeff=0.0):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # blur image
    blur_size = (2 * int(0.005 * min(image.shape)) + 1,) * 2
    image = cv2.GaussianBlur(image, blur_size, 0)
    input_shape = image.shape

    scale_factor = 5000 / image.shape[0]
    output_shape = (np.array(image.shape) * scale_factor).astype(np.int16)
    out_image = np.zeros(output_shape)
    N_lines = 5000
    points_to_sample = []
    image_max = np.max(image)
    while len(points_to_sample) < N_lines:
        point = np.array(
            [
                random.randint(0, input_shape[0] - 1),
                random.randint(0, input_shape[1] - 1),
            ]
        )
        if image[tuple(point)] > 0.1 * image_max:
            points_to_sample.append(point)

    gradient = gradient_of_image(image).astype(np.float32)
    gradient /= np.quantile(gradient, 0.99) / 255.0
    gradient = np.clip(gradient, 0, 255.0)
    # return (gradient[:, :, 0] ** 2 + gradient[:, :, 1] ** 2) ** 0.5

    # max_grad = (
    #     np.quantile((gradient[:, :, 0] ** 2 + gradient[:, :, 1] ** 2), 0.9) ** 0.5
    # )

    line_length = 0.2 * np.min(image.shape)
    for p in points_to_sample:
        n = gradient[tuple(p)].astype(np.float32)
        n_len = (n[0] ** 2 + n[1] ** 2) ** 0.5
        if n_len < 0.0001:
            continue
        n_perpendicular = np.array([-n[1], n[0]]) / n_len
        # add randomness to the angle of the tangent line
        # if random_coeff > 0.001:
        #     n_perpendicular = (
        #         n_perpendicular
        #         + (np.random.random((2,)) - np.array([0.5, 0.5])) * random_coeff
        #     )

        strength = image[tuple(p)] / 255.0
        # strength = min(n_len / max_grad, 1.0)
        color = np.array([255, 255, 255]) * strength
        thickness = 1
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
            int(color[0]),
            thickness,
        )

    return out_image
