import os
from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, List, Optional, Set, Tuple

import cv2
import numpy as np

from event import normalize_object_type

STRONG_REID_THRESHOLD = 0.80
MAX_TIME_DIFF_SECONDS = 10.0
SOFT_MATCH_THRESHOLD = 0.45


def _normalize_embedding(vector: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vector)
    if norm == 0:
        return vector.astype(np.float32, copy=False)
    return (vector / norm).astype(np.float32, copy=False)


def _clip_bbox(frame_shape, bbox: Tuple[int, int, int, int]) -> Tuple[int, int, int, int]:
    height, width = frame_shape[:2]
    x1, y1, x2, y2 = [int(value) for value in bbox]

    x1 = max(0, min(x1, width - 1))
    y1 = max(0, min(y1, height - 1))
    x2 = max(x1 + 1, min(x2, width))
    y2 = max(y1 + 1, min(y2, height))
    return x1, y1, x2, y2


def _bbox_iou(left: Tuple[int, int, int, int], right: Tuple[int, int, int, int]) -> float:
    left_x1, left_y1, left_x2, left_y2 = left
    right_x1, right_y1, right_x2, right_y2 = right

    inter_x1 = max(left_x1, right_x1)
    inter_y1 = max(left_y1, right_y1)
    inter_x2 = min(left_x2, right_x2)
    inter_y2 = min(left_y2, right_y2)

    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    intersection = inter_w * inter_h
    if intersection <= 0:
        return 0.0

    left_area = max(1, (left_x2 - left_x1) * (left_y2 - left_y1))
    right_area = max(1, (right_x2 - right_x1) * (right_y2 - right_y1))
    union = left_area + right_area - intersection
    if union <= 0:
        return 0.0
    return float(intersection / union)


def _extract_fallback_embedding(frame, bbox: Tuple[int, int, int, int]) -> np.ndarray:
    x1, y1, x2, y2 = _clip_bbox(frame.shape, bbox)
    frame_height, frame_width = frame.shape[:2]
    box_width = max(1, x2 - x1)
    box_height = max(1, y2 - y1)
    frame_area = max(1, frame_width * frame_height)

    geometry_features = np.array(
        [
            box_width / max(1.0, float(frame_width)),
            box_height / max(1.0, float(frame_height)),
            (box_width * box_height) / float(frame_area),
            min((box_width / max(1.0, float(box_height))) / 4.0, 1.0),
        ],
        dtype=np.float32,
    )

    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        histogram = np.zeros(48, dtype=np.float32)
        edge_hist = np.zeros(32, dtype=np.float32)
    else:
        resized = cv2.resize(crop, (64, 128), interpolation=cv2.INTER_AREA)
        hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
        hist_h = cv2.calcHist([hsv], [0], None, [16], [0, 180]).flatten()
        hist_s = cv2.calcHist([hsv], [1], None, [16], [0, 256]).flatten()
        hist_v = cv2.calcHist([hsv], [2], None, [16], [0, 256]).flatten()
        histogram = np.concatenate(
            [
                hist_h.astype(np.float32),
                hist_s.astype(np.float32),
                hist_v.astype(np.float32),
            ]
        )

        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 100, 200)
        edge_hist = cv2.calcHist([edges], [0], None, [32], [0, 256]).flatten().astype(np.float32)

    embedding = np.concatenate([histogram, edge_hist, geometry_features * 4.0])
    return _normalize_embedding(embedding)


def extract_detection_features(frame, bbox: Tuple[int, int, int, int]) -> np.ndarray:
    return _extract_fallback_embedding(frame, bbox)


def extract_color_histogram(frame, bbox: Tuple[int, int, int, int]) -> Optional[np.ndarray]:
    return extract_shirt_color(frame, bbox)


def extract_shirt_color(frame, bbox: Tuple[int, int, int, int]) -> Optional[np.ndarray]:
    x1, y1, x2, y2 = _clip_bbox(frame.shape, bbox)
    box_height = max(1, y2 - y1)
    upper_y1 = y1 + int(box_height * 0.18)
    upper_y2 = y1 + int(box_height * 0.55)
    upper_y2 = max(upper_y1 + 1, min(upper_y2, y2))

    crop = frame[upper_y1:upper_y2, x1:x2]
    if crop.size == 0:
        return None

    rgb_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    return np.mean(rgb_crop.reshape(-1, 3), axis=0).astype(np.float32)


def color_distance(left: Optional[np.ndarray], right: Optional[np.ndarray]) -> float:
    if left is None or right is None:
        return 0.0
    if left.size != 3 or right.size != 3:
        return 0.0
    max_distance = float(np.sqrt(3 * (255.0 ** 2)))
    return min(float(np.linalg.norm(left.astype(np.float32) - right.astype(np.float32)) / max_distance), 1.0)


def color_similarity(left: Optional[np.ndarray], right: Optional[np.ndarray]) -> float:
    if left is None or right is None:
        return 0.0
    return 1.0 - color_distance(left, right)


def cosine_similarity(left: np.ndarray, right: np.ndarray) -> float:
    if left.size == 0 or right.size == 0:
        return 0.0
    return float(np.dot(left, right))


class FastReIDEmbedder:
    def __init__(self):
        self._predictor = None
        self.backend_name = "fallback"
        self.embedding_size = 84
        self._build_backend()

    def _build_backend(self) -> None:
        config_path = os.getenv("FASTREID_CONFIG")
        weights_path = os.getenv("FASTREID_WEIGHTS")
        if not config_path or not weights_path:
            return
        if not os.path.exists(config_path) or not os.path.exists(weights_path):
            return

        try:
            from fastreid.config import get_cfg
            from fastreid.engine.defaults import DefaultPredictor
        except ImportError:
            return

        device = os.getenv("FASTREID_DEVICE")
        if not device:
            device = "cpu"
            try:
                import torch

                if torch.cuda.is_available():
                    device = "cuda"
            except ImportError:
                device = "cpu"

        cfg = get_cfg()
        cfg.merge_from_file(config_path)
        cfg.MODEL.WEIGHTS = weights_path
        if hasattr(cfg.MODEL, "DEVICE"):
            cfg.MODEL.DEVICE = device

        self._predictor = DefaultPredictor(cfg)
        self.backend_name = "fastreid"

    def extract(self, frame, bbox: Tuple[int, int, int, int]) -> np.ndarray:
        if self._predictor is None:
            return _extract_fallback_embedding(frame, bbox)

        x1, y1, x2, y2 = _clip_bbox(frame.shape, bbox)
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return _extract_fallback_embedding(frame, bbox)

        try:
            features = self._predictor(crop)
            if hasattr(features, "detach"):
                features = features.detach()
            if hasattr(features, "cpu"):
                features = features.cpu()
            vector = np.asarray(features, dtype=np.float32).reshape(-1)
            if vector.size == 0:
                return _extract_fallback_embedding(frame, bbox)
            self.embedding_size = int(vector.size)
            return _normalize_embedding(vector)
        except Exception:
            self._predictor = None
            self.backend_name = "fallback"
            return _extract_fallback_embedding(frame, bbox)


@dataclass
class IdentityRecord:
    global_id: int
    object_type: str
    embedding: np.ndarray
    embedding_history: Deque[np.ndarray]
    shirt_color: Optional[np.ndarray]
    color_history: Deque[np.ndarray]
    last_camera_id: int
    last_seen_time: float
    camera_history: List[int]


@dataclass
class EmbeddingCacheRecord:
    bbox: Tuple[int, int, int, int]
    embedding: np.ndarray
    last_seen_time: float


class GlobalIdentityManager:
    def __init__(
        self,
        similarity_threshold: float = 0.80,
        spatial_threshold: float = 150.0,
        cross_camera_similarity_threshold: Optional[float] = None,
        match_window_seconds: float = 10.0,
        embedding_cache_ttl_seconds: float = 0.75,
        embedding_cache_iou_threshold: float = 0.85,
        score_threshold: float = 0.45,
        embedding_memory_size: int = 10,
    ):
        self.similarity_threshold = STRONG_REID_THRESHOLD
        self.spatial_threshold = spatial_threshold
        self.cross_camera_similarity_threshold = STRONG_REID_THRESHOLD
        self.match_window_seconds = MAX_TIME_DIFF_SECONDS
        self.embedding_cache_ttl_seconds = embedding_cache_ttl_seconds
        self.embedding_cache_iou_threshold = embedding_cache_iou_threshold
        self.score_threshold = SOFT_MATCH_THRESHOLD
        self.embedding_memory_size = max(1, int(embedding_memory_size))
        self.next_global_id = 1
        self.global_id_map: Dict[int, np.ndarray] = {}
        self.track_id_to_global_id: Dict[Tuple[int, int], int] = {}
        self.local_to_global = self.track_id_to_global_id
        self.identity_store: Dict[int, IdentityRecord] = {}
        self.camera_time_assignments: Dict[Tuple[int, float], Set[int]] = {}
        self.embedding_cache: Dict[Tuple[int, int], EmbeddingCacheRecord] = {}
        self.embedder = FastReIDEmbedder()
        self.embedding_size = self.embedder.embedding_size

    def _color_similarity(
        self,
        left: Optional[np.ndarray],
        right: Optional[np.ndarray],
    ) -> float:
        return color_similarity(left, right)

    def _log_match_attempt(
        self,
        candidate_gid: int,
        reid_similarity: float,
        color_similarity: float,
        time_diff: float,
        final_score: float,
        decision: str,
    ) -> None:
        print(
            "[DEBUG MATCH]\n"
            f"GID: {candidate_gid}\n"
            f"ReID: {reid_similarity:.4f}\n"
            f"Color: {color_similarity:.4f}\n"
            f"Time: {time_diff:.2f}\n"
            f"Score: {final_score:.4f}\n"
            f"Decision: {decision}"
        )

    def _allocate_global_id(self) -> int:
        global_id = self.next_global_id
        self.next_global_id += 1
        return global_id

    def _is_valid_global_id(self, global_id: Optional[int]) -> bool:
        return isinstance(global_id, int) and global_id > 0

    def _get_embedding(
        self,
        camera_id: int,
        track_id: int,
        frame,
        bbox: Tuple[int, int, int, int],
        current_time: float,
    ) -> np.ndarray:
        local_key = (camera_id, track_id)
        cached = self.embedding_cache.get(local_key)
        if cached is not None:
            if (
                abs(float(current_time) - float(cached.last_seen_time)) <= self.embedding_cache_ttl_seconds
                and _bbox_iou(cached.bbox, bbox) >= self.embedding_cache_iou_threshold
            ):
                return cached.embedding

        embedding = self.embedder.extract(frame, bbox)
        self.embedding_size = self.embedder.embedding_size
        self.embedding_cache[local_key] = EmbeddingCacheRecord(
            bbox=bbox,
            embedding=embedding,
            last_seen_time=float(current_time),
        )
        return embedding

    def assign_global_id(
        self,
        camera_id: int,
        track_id: int,
        frame,
        bbox: Tuple[int, int, int, int],
        object_type: str,
        current_time: float,
    ) -> int:
        local_key = (camera_id, track_id)
        object_type = normalize_object_type(object_type)
        embedding = self._get_embedding(camera_id, track_id, frame, bbox, current_time)
        shirt_color = extract_shirt_color(frame, bbox)
        self._prune_runtime_caches(current_time)

        existing_global_id = self.track_id_to_global_id.get(local_key)
        if self._is_valid_global_id(existing_global_id):
            print(
                "[MATCH FOUND] "
                f"GID {existing_global_id} reused for camera {camera_id} track {track_id} "
                "(existing local track)"
            )
            self._refresh_record(
                existing_global_id,
                object_type,
                embedding,
                shirt_color,
                camera_id,
                current_time,
            )
            self._mark_camera_time_assignment(camera_id, current_time, existing_global_id)
            return existing_global_id

        if local_key in self.track_id_to_global_id:
            self.track_id_to_global_id.pop(local_key, None)

        matched_global_id = self._match_existing_identity(
            camera_id,
            object_type,
            embedding,
            shirt_color,
            current_time,
        )

        if not self._is_valid_global_id(matched_global_id):
            matched_global_id = self._allocate_global_id()
            print(
                "[NEW ID CREATED] "
                f"GID {matched_global_id} for camera {camera_id} track {track_id}"
            )

        self._refresh_record(
            matched_global_id,
            object_type,
            embedding,
            shirt_color,
            camera_id,
            current_time,
        )

        self.track_id_to_global_id[local_key] = matched_global_id
        self._mark_camera_time_assignment(camera_id, current_time, matched_global_id)
        return matched_global_id

    def _assignment_slot_key(self, camera_id: int, current_time: float) -> Tuple[int, float]:
        return camera_id, round(float(current_time), 3)

    def _mark_camera_time_assignment(self, camera_id: int, current_time: float, global_id: int) -> None:
        slot_key = self._assignment_slot_key(camera_id, current_time)
        assigned_global_ids = self.camera_time_assignments.setdefault(slot_key, set())
        assigned_global_ids.add(global_id)

    def _prune_runtime_caches(self, current_time: float) -> None:
        stale_assignment_keys = [
            slot_key
            for slot_key in self.camera_time_assignments
            if abs(slot_key[1] - float(current_time)) >= self.match_window_seconds
        ]
        for slot_key in stale_assignment_keys:
            self.camera_time_assignments.pop(slot_key, None)

        stale_embedding_keys = [
            cache_key
            for cache_key, record in self.embedding_cache.items()
            if abs(float(record.last_seen_time) - float(current_time))
            > max(self.match_window_seconds, self.embedding_cache_ttl_seconds)
        ]
        for cache_key in stale_embedding_keys:
            self.embedding_cache.pop(cache_key, None)

    def _match_existing_identity(
        self,
        camera_id: int,
        object_type: str,
        embedding: np.ndarray,
        shirt_color: Optional[np.ndarray],
        current_time: float,
    ) -> Optional[int]:
        object_type = normalize_object_type(object_type)

        best_global_id = None
        best_score = float("-inf")
        best_time_diff = float("inf")
        assigned_in_same_camera_slot = self.camera_time_assignments.get(
            self._assignment_slot_key(camera_id, current_time),
            set(),
        )

        for global_id, record in self.identity_store.items():
            if record.object_type != object_type:
                continue

            if global_id in assigned_in_same_camera_slot:
                continue

            reid_similarity = cosine_similarity(embedding, record.embedding)
            color_similarity = self._color_similarity(shirt_color, record.shirt_color)
            time_diff = float(current_time) - float(record.last_seen_time)

            if time_diff < 0 or time_diff >= self.match_window_seconds:
                self._log_match_attempt(
                    candidate_gid=global_id,
                    reid_similarity=reid_similarity,
                    color_similarity=color_similarity,
                    time_diff=abs(time_diff),
                    final_score=float("-inf"),
                    decision="new_id(candidate_filter)",
                )
                continue

            if reid_similarity > STRONG_REID_THRESHOLD:
                self._log_match_attempt(
                    candidate_gid=global_id,
                    reid_similarity=reid_similarity,
                    color_similarity=color_similarity,
                    time_diff=abs(time_diff),
                    final_score=reid_similarity,
                    decision="match(stage1_strong_reid)",
                )
                print(f"[STRONG MATCH] GID {global_id}")
                print(f"[MATCH FOUND] GID {global_id}")
                return global_id

            time_penalty = min(abs(time_diff) / MAX_TIME_DIFF_SECONDS, 1.0)
            score = (
                0.7 * reid_similarity
                + 0.05 * color_similarity
                - 0.2 * time_penalty
            )

            decision = "match(stage2_soft)" if score > self.score_threshold else "new_id"
            self._log_match_attempt(
                candidate_gid=global_id,
                reid_similarity=reid_similarity,
                color_similarity=color_similarity,
                time_diff=abs(time_diff),
                final_score=score,
                decision=decision,
            )

            if score <= self.score_threshold:
                continue

            if score < best_score:
                continue

            if score == best_score and time_diff >= best_time_diff:
                continue

            best_score = score
            best_time_diff = time_diff
            best_global_id = global_id

        if best_global_id is not None:
            print(f"[MATCH FOUND] GID {best_global_id}")

        return best_global_id

    def _refresh_record(
        self,
        global_id: int,
        object_type: str,
        embedding: np.ndarray,
        shirt_color: Optional[np.ndarray],
        camera_id: int,
        current_time: float,
    ) -> None:
        object_type = normalize_object_type(object_type)
        record = self.identity_store.get(global_id)
        if record is None:
            history = deque([embedding], maxlen=self.embedding_memory_size)
            avg_embedding = _normalize_embedding(np.mean(np.stack(history), axis=0))
            color_history: Deque[np.ndarray] = deque(maxlen=self.embedding_memory_size)
            if shirt_color is not None:
                color_history.append(shirt_color)
            self.global_id_map[global_id] = avg_embedding
            self.identity_store[global_id] = IdentityRecord(
                global_id=global_id,
                object_type=object_type,
                embedding=avg_embedding,
                embedding_history=history,
                shirt_color=self._average_color(color_history),
                color_history=color_history,
                last_camera_id=camera_id,
                last_seen_time=float(current_time),
                camera_history=[camera_id],
            )
            return

        record.embedding_history.append(embedding)
        avg_embedding = _normalize_embedding(np.mean(np.stack(record.embedding_history), axis=0))
        record.embedding = avg_embedding
        self.global_id_map[global_id] = avg_embedding
        record.object_type = object_type
        if shirt_color is not None:
            record.color_history.append(shirt_color)
            record.shirt_color = self._average_color(record.color_history)
        record.last_camera_id = camera_id
        record.last_seen_time = float(current_time)
        record.camera_history.append(camera_id)
        if len(record.camera_history) > self.embedding_memory_size:
            record.camera_history = record.camera_history[-self.embedding_memory_size:]

    def _average_color(self, color_history: Deque[np.ndarray]) -> Optional[np.ndarray]:
        if not color_history:
            return None
        return np.mean(np.stack(color_history), axis=0).astype(np.float32)

    def clear_camera_track_mappings(self, camera_id: Optional[int] = None) -> None:
        if camera_id is None:
            self.track_id_to_global_id.clear()
            self.camera_time_assignments.clear()
            self.embedding_cache.clear()
            return

        stale_track_keys = [key for key in self.track_id_to_global_id if key[0] == camera_id]
        for key in stale_track_keys:
            self.track_id_to_global_id.pop(key, None)

        stale_assignment_keys = [key for key in self.camera_time_assignments if key[0] == camera_id]
        for key in stale_assignment_keys:
            self.camera_time_assignments.pop(key, None)

        stale_embedding_keys = [key for key in self.embedding_cache if key[0] == camera_id]
        for key in stale_embedding_keys:
            self.embedding_cache.pop(key, None)
