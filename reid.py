from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import cv2
import numpy as np

from event import normalize_object_type


def _normalize_embedding(vector: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vector)
    if norm == 0:
        return vector
    return vector / norm


def extract_embedding(frame, bbox: Tuple[int, int, int, int]) -> Optional[np.ndarray]:
    x1, y1, x2, y2 = bbox
    height, width = frame.shape[:2]

    x1 = max(0, min(x1, width - 1))
    y1 = max(0, min(y1, height - 1))
    x2 = max(x1 + 1, min(x2, width))
    y2 = max(y1 + 1, min(y2, height))

    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return None

    resized = cv2.resize(crop, (64, 128), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

    hist_h = cv2.calcHist([hsv], [0], None, [16], [0, 180]).flatten()
    hist_s = cv2.calcHist([hsv], [1], None, [8], [0, 256]).flatten()
    hist_v = cv2.calcHist([hsv], [2], None, [8], [0, 256]).flatten()

    vertical_profile = gray.mean(axis=1)
    horizontal_profile = gray.mean(axis=0)

    coarse_vertical = np.array_split(vertical_profile, 8)
    coarse_horizontal = np.array_split(horizontal_profile, 4)

    profile_features = np.array(
        [segment.mean() for segment in coarse_vertical]
        + [segment.mean() for segment in coarse_horizontal],
        dtype=np.float32,
    )

    aspect_ratio = np.array(
        [(x2 - x1) / max(1.0, float(y2 - y1))],
        dtype=np.float32,
    )

    embedding = np.concatenate(
        [
            hist_h.astype(np.float32),
            hist_s.astype(np.float32),
            hist_v.astype(np.float32),
            profile_features,
            aspect_ratio,
        ]
    )

    return _normalize_embedding(embedding)


def cosine_similarity(left: np.ndarray, right: np.ndarray) -> float:
    if left.size == 0 or right.size == 0:
        return 0.0
    return float(np.dot(left, right))


@dataclass
class IdentityRecord:
    global_id: int
    object_type: str
    embedding: np.ndarray
    last_camera_id: int
    last_frame_number: int


class GlobalIdentityManager:
    def __init__(self, similarity_threshold: float = 0.88):
        self.similarity_threshold = similarity_threshold
        self.next_global_id = 1
        self.local_to_global: Dict[Tuple[int, int], int] = {}
        self.identity_store: Dict[int, IdentityRecord] = {}
        self.embedding_size = 45

    def assign_global_id(
        self,
        camera_id: int,
        track_id: int,
        frame,
        bbox: Tuple[int, int, int, int],
        object_type: str,
        frame_number: int,
    ) -> int:
        local_key = (camera_id, track_id)
        object_type = normalize_object_type(object_type)
        embedding = extract_embedding(frame, bbox)

        existing_global_id = self.local_to_global.get(local_key)
        if existing_global_id is not None:
            self._refresh_record(
                existing_global_id,
                object_type,
                embedding,
                camera_id,
                frame_number,
            )
            return existing_global_id

        matched_global_id = self._match_existing_identity(
            camera_id,
            object_type,
            embedding,
        )

        if matched_global_id is None:
            matched_global_id = self.next_global_id
            self.next_global_id += 1
            if embedding is None:
                embedding = np.zeros(self.embedding_size, dtype=np.float32)
            self.identity_store[matched_global_id] = IdentityRecord(
                global_id=matched_global_id,
                object_type=object_type,
                embedding=embedding,
                last_camera_id=camera_id,
                last_frame_number=frame_number,
            )
        else:
            self._refresh_record(
                matched_global_id,
                object_type,
                embedding,
                camera_id,
                frame_number,
            )

        self.local_to_global[local_key] = matched_global_id
        return matched_global_id

    def _match_existing_identity(
        self,
        camera_id: int,
        object_type: str,
        embedding: Optional[np.ndarray],
    ) -> Optional[int]:
        object_type = normalize_object_type(object_type)
        if embedding is None:
            return None

        best_global_id = None
        best_similarity = self.similarity_threshold

        for global_id, record in self.identity_store.items():
            if record.object_type != object_type:
                continue

            similarity = cosine_similarity(embedding, record.embedding)
            if record.last_camera_id == camera_id:
                similarity -= 0.02

            if similarity > best_similarity:
                best_similarity = similarity
                best_global_id = global_id

        return best_global_id

    def _refresh_record(
        self,
        global_id: int,
        object_type: str,
        embedding: Optional[np.ndarray],
        camera_id: int,
        frame_number: int,
    ) -> None:
        object_type = normalize_object_type(object_type)
        record = self.identity_store.get(global_id)
        if record is None:
            if embedding is None:
                embedding = np.zeros(self.embedding_size, dtype=np.float32)
            self.identity_store[global_id] = IdentityRecord(
                global_id=global_id,
                object_type=object_type,
                embedding=embedding,
                last_camera_id=camera_id,
                last_frame_number=frame_number,
            )
            return

        if embedding is not None:
            record.embedding = _normalize_embedding((record.embedding * 0.7) + (embedding * 0.3))

        record.object_type = object_type
        record.last_camera_id = camera_id
        record.last_frame_number = frame_number
