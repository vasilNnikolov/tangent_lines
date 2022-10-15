import cv2
from tqdm import tqdm

import filter

input_filename = "face.webm"

vidcap = cv2.VideoCapture(input_filename)
success = True

frame_count = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = vidcap.get(cv2.CAP_PROP_FPS)
input_width = int(vidcap.get(cv2.CAP_PROP_FRAME_WIDTH))
input_height = int(vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT))

scaling_coefficient = 2
output_width, output_height = (
    int(scaling_coefficient * input_width),
    int(scaling_coefficient * input_height),
)
out = cv2.VideoWriter(
    "output_face.avi",
    cv2.VideoWriter_fourcc(*"MJPG"),
    fps,
    (output_width, output_height),
)
# for i in tqdm(range(1)):
for i in tqdm(range(frame_count)):
    success, image = vidcap.read()
    if not success:
        print(f"error on frame {i}")
    filtered_im = filter.filter_image(image, scale_factor=scaling_coefficient)
    transformed_image = cv2.cvtColor(filtered_im, cv2.COLOR_GRAY2BGR)
    out.write(transformed_image)

out.release()
