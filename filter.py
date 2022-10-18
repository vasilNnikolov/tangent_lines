import cv2
import matplotlib.pyplot as plt
import numpy as np

import timing


def gradient_of_image(image):
    image = image.astype(np.float32)
    ksize = 5
    gX = cv2.Sobel(image, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=ksize).astype(np.float32)
    gY = cv2.Sobel(image, ddepth=cv2.CV_32F, dx=0, dy=1, ksize=ksize).astype(np.float32)
    # tries to kind of normalise the gradient, since brighter areas have higher gradient
    offset = 10
    return np.stack([gY / (image + offset), gX / (image + offset)], axis=2)


def filter_image(image: np.ndarray, timer: timing.Timer, scale_factor=4) -> np.ndarray:
    timer.add_checkpoint("cvt_image_to_gray")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # blur image
    timer.add_checkpoint("blur")
    blur_size = (2 * int(0.007 * min(image.shape)) + 1,) * 2
    image = cv2.GaussianBlur(image, blur_size, 0)

    timer.add_checkpoint("compute_gradient")
    gradient = gradient_of_image(image)

    timer.add_checkpoint("compute gradient magnitude")
    gradient_magnitude = (gradient[:, :, 0] ** 2 + gradient[:, :, 1] ** 2) ** 0.5

    timer.add_checkpoint("find quantile and get grid")
    # edge_threshold = np.quantile(gradient_magnitude, 0.8)
    edge_threshold = np.quantile(
        np.random.choice(
            gradient_magnitude.flatten(),
            size=int(0.1 * (image.shape[0] * image.shape[1])),
        ),
        0.8,
    )

    timer.add_checkpoint("pick strongest edges")
    edge_coords = np.moveaxis(np.mgrid[0 : image.shape[0], 0 : image.shape[1]], 0, -1)[
        gradient_magnitude > edge_threshold
    ]

    # the number of lines that will be drawn
    timer.add_checkpoint("pick random edges")
    N = min(len(edge_coords), 10000)
    edge_coords = edge_coords[np.random.randint(0, len(edge_coords), N)]

    # from this point on, we only care about coordinates which are in edge_coords
    edge_gradients = gradient[tuple(edge_coords.T)]
    points_gradient_magnitudes = gradient_magnitude[tuple(edge_coords.T)]

    timer.add_checkpoint("normalise picked gradients")
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

    timer.add_checkpoint("draw lines")
    output_shape = (np.array(image.shape) * scale_factor).astype(np.int16)
    out_image = np.zeros(tuple(output_shape)).astype(np.uint8)

    # drawing parameters
    line_length = 0.02 * int((image.shape[0] * image.shape[1]) ** 0.5)
    line_thickness = min(1, output_shape[0] // 500)

    # arrays holding the starts and ends of the lines we need to draw
    # shape (N, 2)
    line_starts = np.flip(
        scale_factor * (edge_coords - normalised_perpendicular_vectors * line_length),
        axis=1,
    ).astype(np.int16)
    line_ends = np.flip(
        scale_factor * (edge_coords + normalised_perpendicular_vectors * line_length),
        axis=1,
    ).astype(np.int16)

    for index in range(len(edge_coords)):
        color = int((normalised_gradient_magnitudes[index] ** 0.5) * 255)
        cv2.line(
            out_image,
            line_starts[index],
            line_ends[index],
            color,
            line_thickness,
        )

    timer.add_checkpoint("end")

    return out_image
