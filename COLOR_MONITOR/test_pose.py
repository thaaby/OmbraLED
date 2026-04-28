import urllib.request
import os
import sys

url = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
if not os.path.exists("pose_landmarker_lite.task"):
    import ssl
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    with urllib.request.urlopen(req, context=ctx) as response, open("pose_landmarker_lite.task", 'wb') as out_file:
        out_file.write(response.read())

import mediapipe as mp
import cv2
import numpy as np

BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path='pose_landmarker_lite.task'),
    running_mode=VisionRunningMode.IMAGE,
    num_poses=2,
    output_segmentation_masks=True
)

landmarker = PoseLandmarker.create_from_options(options)

# create a dummy image
img = np.zeros((480, 640, 3), dtype=np.uint8)
# Add some fake "persons" shapes maybe, but just empty is fine; it should return empty list or None
mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=img)
res = landmarker.detect(mp_img)

if res.segmentation_masks is not None:
    print(f"Masks returned: {len(res.segmentation_masks)}")
else:
    print("Masks is None or empty list.")
