import cv2
from tqdm import tqdm

import filter
import timing

input_filename = "lozenetz.mp4"

vidcap = cv2.VideoCapture(input_filename)
print(input_filename)
success = True

frame_count = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = vidcap.get(cv2.CAP_PROP_FPS)
input_width = int(vidcap.get(cv2.CAP_PROP_FRAME_WIDTH))
input_height = int(vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(input_height, input_width)

scaling_coefficient = 1
output_width, output_height = (
    int(scaling_coefficient * input_width),
    int(scaling_coefficient * input_height),
)
out = cv2.VideoWriter(
    "out_lozen_testing.avi",
    cv2.VideoWriter_fourcc(*"MJPG"),
    fps,
    (output_width, output_height),
)

timer = timing.Timer()
for i in tqdm(range(50)):
    success, image = vidcap.read()
    if not success:
        print(f"error on frame {i}")
    filtered_im = filter.filter_image(image, timer, scale_factor=scaling_coefficient)
    transformed_image = cv2.cvtColor(filtered_im, cv2.COLOR_GRAY2BGR)
    out.write(transformed_image)

timer.statistics()
out.release()
