import os
from dataclasses import dataclass
from typing import Dict, Optional, Set, Tuple

import cv2
import numpy as np

from event import normalize_object_type


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
    return extract_detection_features(frame, bbox)


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
    last_camera_id: int
    last_seen_time: float


@dataclass
class EmbeddingCacheRecord:
    bbox: Tuple[int, int, int, int]
    embedding: np.ndarray
    last_seen_time: float


class GlobalIdentityManager:
    def __init__(
        self,
        similarity_threshold: float = 0.75,
        spatial_threshold: float = 150.0,
        cross_camera_similarity_threshold: Optional[float] = None,
        match_window_seconds: float = 3.0,
        embedding_cache_ttl_seconds: float = 0.75,
        embedding_cache_iou_threshold: float = 0.85,
    ):
        self.similarity_threshold = similarity_threshold
        self.spatial_threshold = spatial_threshold
        self.cross_camera_similarity_threshold = (
            cross_camera_similarity_threshold
            if cross_camera_similarity_threshold is not None
            else similarity_threshold
        )
        self.match_window_seconds = match_window_seconds
        self.embedding_cache_ttl_seconds = embedding_cache_ttl_seconds
        self.embedding_cache_iou_threshold = embedding_cache_iou_threshold
        self.next_global_id = 1
        self.global_id_map: Dict[int, np.ndarray] = {}
        self.track_id_to_global_id: Dict[Tuple[int, int], int] = {}
        self.local_to_global = self.track_id_to_global_id
        self.identity_store: Dict[int, IdentityRecord] = {}
        self.camera_time_assignments: Dict[Tuple[int, float], Set[int]] = {}
        self.embedding_cache: Dict[Tuple[int, int], EmbeddingCacheRecord] = {}
        self.embedder = FastReIDEmbedder()
        self.embedding_size = self.embedder.embedding_size

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
        self._prune_runtime_caches(current_time)

        existing_global_id = self.track_id_to_global_id.get(local_key)
        if self._is_valid_global_id(existing_global_id):
            self._refresh_record(
                existing_global_id,
                object_type,
                embedding,
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
            current_time,
        )

        if not self._is_valid_global_id(matched_global_id):
            matched_global_id = self._allocate_global_id()

        self._refresh_record(
            matched_global_id,
            object_type,
            embedding,
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
        current_time: float,
    ) -> Optional[int]:
        object_type = normalize_object_type(object_type)

        best_global_id = None
        best_similarity = 0.0
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

            similarity = cosine_similarity(embedding, record.embedding)
            time_diff = abs(float(current_time) - float(record.last_seen_time))
            threshold = (
                self.cross_camera_similarity_threshold
                if record.last_camera_id != camera_id
                else self.similarity_threshold
            )

            if similarity <= threshold or time_diff >= self.match_window_seconds:
                continue

            if similarity < best_similarity:
                continue

            if similarity == best_similarity and time_diff >= best_time_diff:
                continue

            best_similarity = similarity
            best_time_diff = time_diff
            best_global_id = global_id

        return best_global_id

    def _refresh_record(
        self,
        global_id: int,
        object_type: str,
        embedding: np.ndarray,
        camera_id: int,
        current_time: float,
    ) -> None:
        object_type = normalize_object_type(object_type)
        record = self.identity_store.get(global_id)
        if record is None:
            self.global_id_map[global_id] = embedding
            self.identity_store[global_id] = IdentityRecord(
                global_id=global_id,
                object_type=object_type,
                embedding=embedding,
                last_camera_id=camera_id,
                last_seen_time=float(current_time),
            )
            return

        updated_embedding = _normalize_embedding((record.embedding * 0.7) + (embedding * 0.3))
        record.embedding = updated_embedding
        self.global_id_map[global_id] = updated_embedding
        record.object_type = object_type
        record.last_camera_id = camera_id
        record.last_seen_time = float(current_time)

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
