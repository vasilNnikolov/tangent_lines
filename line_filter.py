import cv2
import numpy as np
from tqdm import tqdm

import timing


def gradient_of_image(image):
    image = image.astype(np.float32)
    ksize = 5
    gX = cv2.Sobel(image, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=ksize).astype(np.float32)
    gY = cv2.Sobel(image, ddepth=cv2.CV_32F, dx=0, dy=1, ksize=ksize).astype(np.float32)
    # tries to kind of normalise the gradient, since brighter areas have higher gradient
    offset = 10
    return np.stack([gY / (image + offset), gX / (image + offset)], axis=2)


def filter_video(input_filename, output_filename):
    """
    filters the video and saves it in a filename
    """
    vidcap = cv2.VideoCapture(input_filename)
    print(input_filename)
    success = True

    frame_count = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = vidcap.get(cv2.CAP_PROP_FPS)
    input_width = int(vidcap.get(cv2.CAP_PROP_FRAME_WIDTH))
    input_height = int(vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    scaling_coefficient = 1
    output_width, output_height = (
        int(scaling_coefficient * input_width),
        int(scaling_coefficient * input_height),
    )
    out = cv2.VideoWriter(
        output_filename,
        cv2.VideoWriter_fourcc(*"MJPG"),
        fps,
        (output_width, output_height),
    )

    timer = timing.Timer()
    # optimisation so no need to run it for every image
    coord_grid = np.moveaxis(np.mgrid[0:input_height, 0:input_width], 0, -1)
    for i in tqdm(range(50)):
        success, image = vidcap.read()
        if not success:
            print(f"error reading on frame {i}")

        filtered_im = filter_image(
            image, timer, scale_factor=scaling_coefficient, coord_grid=coord_grid
        )
        transformed_image = cv2.cvtColor(filtered_im, cv2.COLOR_GRAY2BGR)
        out.write(transformed_image)

    timer.statistics()
    out.release()


def filter_image(
    image: np.ndarray, timer: timing.Timer, scale_factor=4, coord_grid=None
) -> np.ndarray:
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

    timer.add_checkpoint("find quantile")
    edge_threshold = np.quantile(
        np.random.choice(
            gradient_magnitude.flatten(),
            size=int(0.1 * image.shape[0] * image.shape[1]),
        ),
        0.8,
    )

    timer.add_checkpoint("get grid")
    if coord_grid is None:
        coord_grid = np.moveaxis(
            np.mgrid[0 : image.shape[0], 0 : image.shape[1]], 0, -1
        )

    timer.add_checkpoint("pick strongest edges")
    edge_coords = coord_grid[gradient_magnitude > edge_threshold]

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
