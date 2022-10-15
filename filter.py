import cv2
import matplotlib.pyplot as plt
import numpy as np


def gradient_of_image(image):
    image = image.astype(np.float32)
    ksize = 7
    # gX = cv2.Scharr(image, ddepth=cv2.CV_32F, dx=1, dy=0).astype(np.float32)
    # gY = cv2.Scharr(image, ddepth=cv2.CV_32F, dx=0, dy=1).astype(np.float32)
    gX = cv2.Sobel(image, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=ksize).astype(np.float32)
    gY = cv2.Sobel(image, ddepth=cv2.CV_32F, dx=0, dy=1, ksize=ksize).astype(np.float32)
    # tries to kind of normalise the gradient, since brighter areas have higher gradient
    offset = 10
    return np.stack([gY / (image + offset), gX / (image + offset)], axis=2)


def filter_image(image: np.ndarray, scale_factor=4) -> np.ndarray:
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # blur image
    blur_size = (2 * int(0.007 * min(image.shape)) + 1,) * 2
    image = cv2.GaussianBlur(image, blur_size, 0)

    gradient = gradient_of_image(image)
    gradient_magnitude = (gradient[:, :, 0] ** 2 + gradient[:, :, 1] ** 2) ** 0.5

    edge_coords = np.moveaxis(np.mgrid[0 : image.shape[0], 0 : image.shape[1]], 0, -1)[
        gradient_magnitude > np.quantile(gradient_magnitude, 0.8)
    ]
    # the number of lines that will be drawn
    N = min(len(edge_coords), 10000)
    edge_coords = edge_coords[np.random.randint(0, len(edge_coords), N)]

    # from this point on, we only care about coordinates which are in edge_coords
    edge_gradients = gradient[tuple(edge_coords.T)]
    points_gradient_magnitudes = gradient_magnitude[tuple(edge_coords.T)]

    normalised_gradient_magnitudes = points_gradient_magnitudes / np.max(
        points_gradient_magnitudes
    )

    # make their length 1
    normalised_perpendicular_vectors = (
        edge_gradients / points_gradient_magnitudes[:, np.newaxis]
    )

    # turn them 90 degrees
    normalised_perpendicular_vectors = np.stack(
        [
            normalised_perpendicular_vectors[:, 1],
            -normalised_perpendicular_vectors[:, 0],
        ],
        axis=1,
    )

    output_shape = (np.array(image.shape) * scale_factor).astype(np.int16)
    out_image = np.zeros(tuple(output_shape)).astype(np.uint8)

    # drawing parameters
    line_length = 0.03 * np.min(image.shape)
    line_thickness = min(1, output_shape[0] // 500)

    # line drawing
    for index, p in enumerate(edge_coords):
        n_perpendicular = normalised_perpendicular_vectors[index]

        color = int((normalised_gradient_magnitudes[index] ** 0.5) * 255)
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
