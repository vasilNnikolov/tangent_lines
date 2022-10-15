import random
import sys

import cv2
import matplotlib.pyplot as plt
import numpy as np


def compute_edges(image: np.ndarray, lower_bound=15, upper_bound=26) -> np.ndarray:
    return image
    # kernel_size_1 = max(5, 2 * int(0.0002 * image.shape[0]) + 1)
    # kernel_size_2 = max(5, 2 * int(0.01 * image.shape[0]) + 1)
    # blurred_image = cv2.GaussianBlur(image, (kernel_size_1, kernel_size_1), 0)
    # plt.imshow(blurred_image)
    # plt.show()
    # return cv2.GaussianBlur(
    #     cv2.Canny(
    #         blurred_image,
    #         lower_bound,
    #         upper_bound,
    #     ),
    #     (kernel_size_2, kernel_size_2),
    #     0,
    # )


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


def filter_image(image: np.ndarray):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    input_shape = image.shape

    k = 15
    edges = compute_edges(image, k, 2 * k)
    # plt.imshow(edges)
    # plt.show()

    scale_factor = 5000 / image.shape[0]
    output_shape = (np.array(image.shape) * scale_factor).astype(np.int16)
    out_image = np.zeros(output_shape)
    N_lines = 50000
    points_to_sample = []
    edge_max = np.max(edges)
    while len(points_to_sample) < N_lines:
        point = np.array(
            [
                random.randint(0, input_shape[0] - 1),
                random.randint(0, input_shape[1] - 1),
            ]
        )
        magnitude = edges[tuple(point)]
        if magnitude > 0.3 * edge_max:
            points_to_sample.append(point)

    gradient = gradient_of_image(image)
    max_grad = np.max((gradient[:, :, 0] ** 2 + gradient[:, :, 1] ** 2)) ** 0.5

    line_length = 0.1 * input_shape[0]
    for i, p in enumerate(points_to_sample):
        # if i % (N_lines // 20) == 0:
        #     print("progress: ", i)
        n = gradient[tuple(p)].astype(np.float32)
        n_len = (n[0] ** 2 + n[1] ** 2) ** 0.5
        if n_len < 0.000001:
            continue
        n_perpendicular = np.array([-n[1], n[0]]) / (max_grad**0.7 * n_len**0.3)
        n_perpendicular = (
            n_perpendicular + (np.random.random((2,)) - np.array([0.5, 0.5])) / 2000
        )

        strength = (image[tuple(p)] / 255) ** 1
        color = np.array([255, 255, 255]) * strength
        thichness = 1
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
            tuple(color),
            thichness,
        )

    return out_image
    # plt.imshow(out_image, cmap="gray")
    # plt.show()
