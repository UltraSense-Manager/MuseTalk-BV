import sys
from face_detection import FaceAlignment,LandmarksType
from os import listdir, path
import subprocess
import numpy as np
import cv2
import pickle
import os
import json
from mmpose.apis import inference_topdown, init_model
from mmpose.structures import merge_data_samples
import torch
from tqdm import tqdm

# initialize the mmpose model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
config_file = './musetalk/utils/dwpose/rtmpose-l_8xb32-270e_coco-ubody-wholebody-384x288.py'
checkpoint_file = './models/dwpose/dw-ll_ucoco_384.pth'
model = init_model(config_file, checkpoint_file, device=device)

# initialize the face detection model
device = "cuda" if torch.cuda.is_available() else "cpu"
fa = FaceAlignment(LandmarksType._2D, flip_input=False,device=device)

# maker if the bbox is not sufficient 
coord_placeholder = (0.0,0.0,0.0,0.0)

def _landmark_batch_size() -> int:
    raw = os.environ.get("LANDMARK_BATCH_SIZE", "1").strip() or "1"
    try:
        return max(1, int(raw))
    except ValueError:
        return 1

def resize_landmark(landmark, w, h, new_w, new_h):
    w_ratio = new_w / w
    h_ratio = new_h / h
    landmark_norm = landmark / [w, h]
    landmark_resized = landmark_norm * [new_w, new_h]
    return landmark_resized

def read_imgs(img_list):
    frames = []
    print('reading images...')
    for img_path in tqdm(img_list):
        frame = cv2.imread(img_path)
        frames.append(frame)
    return frames


def _format_bbox_range(total_frames, upperbondrange, average_range_minus, average_range_plus):
    if not average_range_minus or not average_range_plus:
        return (
            f"Total frame:「{total_frames}」 Manually adjust range : [ n/a ] , "
            f"the current value: {upperbondrange}"
        )
    return (
        f"Total frame:「{total_frames}」 Manually adjust range : [ "
        f"-{int(sum(average_range_minus) / len(average_range_minus))}"
        f"~{int(sum(average_range_plus) / len(average_range_plus))} ] , "
        f"the current value: {upperbondrange}"
    )


def _landmark_bbox_pass(frames, upperbondrange=0, need_coords=True):
    batch_size_fa = _landmark_batch_size()
    if upperbondrange != 0:
        print('get key_landmark and face bounding boxes with the bbox_shift:',upperbondrange)
    else:
        print('get key_landmark and face bounding boxes with the default value')

    coords_list = []
    average_range_minus = []
    average_range_plus = []
    for i in tqdm(range(0, len(frames), batch_size_fa)):
        fb = frames[i:i + batch_size_fa]
        fb_np = np.asarray(fb)
        bbox_batch = fa.get_detections_for_batch(fb_np)
        for j, f in enumerate(bbox_batch):
            frame = fb[j]
            if frame is None:
                if need_coords:
                    coords_list += [coord_placeholder]
                continue
            results = inference_topdown(model, frame)
            results = merge_data_samples(results)
            keypoints = results.pred_instances.keypoints
            face_land_mark = keypoints[0][23:91].astype(np.int32)
            if f is None:  # no face in the image
                if need_coords:
                    coords_list += [coord_placeholder]
                continue

            half_face_coord = face_land_mark[29]
            range_minus = (face_land_mark[30] - face_land_mark[29])[1]
            range_plus = (face_land_mark[29] - face_land_mark[28])[1]
            average_range_minus.append(int(range_minus))
            average_range_plus.append(int(range_plus))
            if upperbondrange != 0:
                half_face_coord[1] = upperbondrange + half_face_coord[1]

            if need_coords:
                half_face_dist = np.max(face_land_mark[:, 1]) - half_face_coord[1]
                min_upper_bond = 0
                upper_bond = max(min_upper_bond, half_face_coord[1] - half_face_dist)
                f_landmark = (
                    np.min(face_land_mark[:, 0]),
                    int(upper_bond),
                    np.max(face_land_mark[:, 0]),
                    np.max(face_land_mark[:, 1]),
                )
                x1, y1, x2, y2 = f_landmark
                if y2 - y1 <= 0 or x2 - x1 <= 0 or x1 < 0:
                    coords_list += [f]
                    print("error bbox:", f)
                else:
                    coords_list += [f_landmark]

    text_range = _format_bbox_range(
        len(frames), upperbondrange, average_range_minus, average_range_plus
    )
    return coords_list, text_range

def get_bbox_range_from_frames(frames, upperbondrange=0):
    """Same as get_bbox_range but uses in-memory BGR frames (OpenCV layout)."""
    _, text_range = _landmark_bbox_pass(frames, upperbondrange, need_coords=False)
    return text_range


def get_bbox_range(img_list, upperbondrange=0):
    frames = read_imgs(img_list)
    return get_bbox_range_from_frames(frames, upperbondrange)


def get_landmark_and_bbox_from_frames(frames, upperbondrange=0):
    """Landmark + bbox extraction using preloaded BGR frames (same as read_imgs output)."""
    coords_list, text_range = _landmark_bbox_pass(frames, upperbondrange, need_coords=True)
    print("********************************************bbox_shift parameter adjustment**********************************************************")
    print(text_range)
    print("*************************************************************************************************************************************")
    return coords_list, frames


def get_landmark_and_bbox_with_range_from_frames(frames, upperbondrange=0):
    coords_list, text_range = _landmark_bbox_pass(frames, upperbondrange, need_coords=True)
    return coords_list, frames, text_range


def get_landmark_and_bbox(img_list, upperbondrange=0):
    frames = read_imgs(img_list)
    return get_landmark_and_bbox_from_frames(frames, upperbondrange)


if __name__ == "__main__":
    img_list = ["./results/lyria/00000.png","./results/lyria/00001.png","./results/lyria/00002.png","./results/lyria/00003.png"]
    crop_coord_path = "./coord_face.pkl"
    coords_list,full_frames = get_landmark_and_bbox(img_list)
    with open(crop_coord_path, 'wb') as f:
        pickle.dump(coords_list, f)
        
    for bbox, frame in zip(coords_list,full_frames):
        if bbox == coord_placeholder:
            continue
        x1, y1, x2, y2 = bbox
        crop_frame = frame[y1:y2, x1:x2]
        print('Cropped shape', crop_frame.shape)
        
        #cv2.imwrite(path.join(save_dir, '{}.png'.format(i)),full_frames[i][0][y1:y2, x1:x2])
    print(coords_list)
