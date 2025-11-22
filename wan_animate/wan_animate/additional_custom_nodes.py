import json
import logging
import numpy as np
import torch
import numpy as np
from einops import rearrange


def extract_best_points_from_pose_part(pose: dict, part: str, top_k: int):    
    points = pose.get(part, [])
    xy_coords = extract_xy_coords_from_points(points)
    most_avg_coords = most_average_points(xy_coords, top_k)
    return most_avg_coords


def extract_xy_coords_from_points(points: list[float]):    
    xy_coords = []
    for i in range(0, len(points), 3):
        if i+2 < len(points):
            x = int(points[i])
            y = int(points[i+1])
            c = int(points[i+2])
            if c != 0:
                xy_coords.append({"x": x, "y": y})
    return xy_coords


def most_average_points(points, k):
    pts = np.array([[p["x"], p["y"]] for p in points])
    center = pts.mean(axis=0)
    dist = np.linalg.norm(pts - center, axis=1)
    idx = np.argsort(dist)[:k]
    return [points[i] for i in idx]



class PoseToSAMPoints:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "pose": ("POSE_KEYPOINT",),
            },
            "optional": {
                "person_index": ("INT", {"default": 0}),
                "top_k_pose": ("INT", {"default": 6}),
                "top_k_face": ("INT", {"default": 1}),
                "top_k_left_hand": ("INT", {"default": 1}),
                "top_k_right_hand": ("INT", {"default": 1}),
            }
        }
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("positive_coords")

    FUNCTION = "convert"
    CATEGORY = "Pose"

    def convert(self, pose, person_index=0, top_k_pose=6, top_k_face=1, top_k_left_hand=1, top_k_right_hand=1):

        # Pose is expected to be a list: [ { "people": [...] }, ... ]
        if not isinstance(pose, list) or len(pose) == 0:
            return ({"positive": [], "negative": [{"x": 0, "y": 0}]},)

        # FIRST FRAME
        frame = pose[0]

        people = frame.get("people", [])
        if not people:
            return ({"positive": [], "negative": [{"x": 0, "y": 0}]},)

        if person_index < 0 or person_index >= len(people):
            person_index = 0
        person = people[person_index]

        most_avg_pose_coords = extract_best_points_from_pose_part(person, "pose_keypoints_2d", top_k_pose)
        most_avg_face_coords = extract_best_points_from_pose_part(person, "face_keypoints_2d", top_k_face)
        most_avg_hand_left_coords = extract_best_points_from_pose_part(person, "hand_left_keypoints_2d", top_k_left_hand)
        most_avg_hand_right_coords = extract_best_points_from_pose_part(person, "hand_right_keypoints_2d", top_k_right_hand)

        points = most_avg_pose_coords + most_avg_face_coords + most_avg_hand_left_coords + most_avg_hand_right_coords

        return (json.dumps(points),)
    

class FaceMaskFromPoseKeypoints:
    @classmethod
    def INPUT_TYPES(s):
        input_types = {
            "required": {
                "pose_kps": ("POSE_KEYPOINT",),
                "person_index": ("INT", {"default": 0, "min": 0, "max": 100, "step": 1, "tooltip": "Index of the person to start with"}),
            }
        }
        return input_types
    RETURN_TYPES = ("MASK",)
    FUNCTION = "createmask"
    CATEGORY = "ControlNet Preprocessors/Pose Keypoint Postprocess"

    def createmask(self, pose_kps, person_index):
        pose_frames = pose_kps
        prev_center = None
        np_frames = []
        for i, pose_frame in enumerate(pose_frames):
            selected_idx, prev_center = self.select_closest_person(pose_frame, person_index if i == 0 else prev_center)
            print(selected_idx, prev_center)
            np_frames.append(self.draw_kps(pose_frame, selected_idx))
        
        if not np_frames:
            # Handle case where no frames were processed
            logging.getLogger().warning("No valid pose frames found, returning empty mask")
            return (torch.zeros((1, 64, 64), dtype=torch.float32),)
            
        np_frames = np.stack(np_frames, axis=0)
        tensor = torch.from_numpy(np_frames).float() / 255.
        print("tensor.shape:", tensor.shape)
        tensor = tensor[:, :, :, 0]
        return (tensor,)

    def select_closest_person(self, pose_frame, prev_center_or_index):
        people = pose_frame["people"]
        if not people:
            return -1, None

        centers = []
        valid_people_indices = []

        for idx, person in enumerate(people):
            if "face_keypoints_2d" not in person or not person["face_keypoints_2d"]:
                continue

            kps = np.array(person["face_keypoints_2d"])
            if len(kps) == 0:
                continue

            n = len(kps) // 3
            if n == 0:
                continue

            facial_kps = rearrange(kps, "(n c) -> n c", n=n, c=3)[:, :2]
            if np.all(facial_kps == 0):
                continue

            center = facial_kps.mean(axis=0)
            if np.isnan(center).any() or np.isinf(center).any():
                continue

            centers.append(center)
            valid_people_indices.append(idx)

        if not centers:
            return -1, None

        # First call: integer = requested index in pose_frame["people"]
        if isinstance(prev_center_or_index, (int, np.integer)):
            requested_idx = int(prev_center_or_index)

            # If the requested person has valid face keypoints in this frame
            if requested_idx in valid_people_indices:
                j = valid_people_indices.index(requested_idx)
                return requested_idx, centers[j]
            else:
                # No usable face for that person in this frame – decide what you want:
                # 1) Return no person:
                return -1, None
                # or 2) If you prefer fallback behavior, uncomment instead:
                # idx = valid_people_indices[0]
                # return idx, centers[0]

        # Subsequent calls: prev_center_or_index is a 2D center
        if prev_center_or_index is not None:
            prev_center = np.array(prev_center_or_index)
            dists = [np.linalg.norm(center - prev_center) for center in centers]
            min_idx = int(np.argmin(dists))
            actual_idx = valid_people_indices[min_idx]
            return actual_idx, centers[min_idx]

        # Fallback
        idx = valid_people_indices[0]
        return idx, centers[0]

    def draw_kps(self, pose_frame, person_index):
        import cv2

        width, height = pose_frame["canvas_width"], pose_frame["canvas_height"]
        canvas = np.zeros((height, width, 3), dtype=np.uint8)
        people = pose_frame["people"]

        if person_index < 0 or person_index >= len(people):
            return canvas

        person = people[person_index]

        if "face_keypoints_2d" not in person or not person["face_keypoints_2d"]:
            return canvas

        face_kps_data = person["face_keypoints_2d"]
        if len(face_kps_data) == 0:
            return canvas

        kps = np.array(face_kps_data, dtype=np.float32)
        n = len(kps) // 3
        if n < 3:
            return canvas  # need at least 3 points for a polygon

        facial_kps = rearrange(kps, "(n c) -> n c", n=n, c=3)[:, :2]

        # keep only non‑zero points
        valid = ~np.all(facial_kps == 0, axis=1)
        facial_kps = facial_kps[valid]
        if len(facial_kps) < 3:
            return canvas

        # basic sanity: drop NaN/Inf and clip
        if np.isnan(facial_kps).any() or np.isinf(facial_kps).any():
            return canvas

        facial_kps[:, 0] = np.clip(facial_kps[:, 0], 0, width - 1)
        facial_kps[:, 1] = np.clip(facial_kps[:, 1], 0, height - 1)
        facial_kps = facial_kps.astype(np.int32)

        # build contour as convex hull of all valid points
        outer_contour = cv2.convexHull(facial_kps)

        if len(outer_contour) >= 3:
            cv2.fillPoly(canvas, pts=[outer_contour], color=(255, 255, 255))

        return canvas
    
NODE_CLASS_MAPPINGS = {
    "PoseToSAMPoints": PoseToSAMPoints,
    "FaceMaskFromPoseKeypoints": FaceMaskFromPoseKeypoints,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PoseToSAMPoints": "PoseToSAMPoints",
    "FaceMaskFromPoseKeypoints": "FaceMaskFromPoseKeypoints",
}
