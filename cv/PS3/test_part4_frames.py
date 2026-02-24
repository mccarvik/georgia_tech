"""
Test find_markers on the frames that produce ps3-4-a-6, ps3-4-b-1, ps3-4-b-2.
Run: python test_part4_frames.py
"""
import os
import cv2
import ps3

VID_DIR = "input_videos"
IMG_DIR = "input_images"
OUT_DIR = "./"

FRAMES_TO_TEST = [
    ("ps3-4-b.mp4", 435, "ps3-4-a-6"),
    ("ps3-4-c.mp4", 47, "ps3-4-b-1"),
    ("ps3-4-c.mp4", 470, "ps3-4-b-2"),
]

def main():
    template = cv2.imread(os.path.join(IMG_DIR, "template.jpg"))
    if template is None:
        print("ERROR: template.jpg not found")
        return
    for video_name, frame_num, out_name in FRAMES_TO_TEST:
        video_path = os.path.join(VID_DIR, video_name)
        if not os.path.exists(video_path):
            print(f"SKIP {out_name}: video not found")
            continue
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num - 1)
        ret, frame = cap.read()
        cap.release()
        if not ret:
            print(f"SKIP {out_name}: could not read frame")
            continue
        markers = ps3.find_markers(frame, template)
        print(f"{out_name}: {markers}")

if __name__ == "__main__":
    main()
