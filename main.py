import cv2
import numpy as np
from tqdm import tqdm

import filter

vidcap = cv2.VideoCapture("vid.mp4")
success = True

frame_count = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
width = int(vidcap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter(
    "filename.avi", cv2.VideoWriter_fourcc(*"MJPG"), 30, (width, height)
)
print(width, height)
for i in tqdm(range(frame_count)):
    success, image = vidcap.read()

    if not success:
        print(f"error on frame {i}")
    filtered_im = cv2.resize(filter.filter_image(image), (width, height)).astype(
        np.uint8
    )
    transformed_image = cv2.cvtColor(filtered_im, cv2.COLOR_GRAY2BGR)
    out.write(transformed_image)

out.release()
