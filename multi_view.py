import math
from typing import Dict, Iterable, Optional, Tuple

import cv2
import numpy as np

DEFAULT_TILE_SIZE = (480, 270)
GRID_BACKGROUND = (18, 18, 18)
PANEL_BACKGROUND = (32, 32, 32)
TEXT_COLOR = (235, 235, 235)
HIGHLIGHT_COLOR = (0, 0, 255)
FRAME_COLOR = (80, 80, 80)


def _fit_with_padding(frame: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    target_width, target_height = size
    canvas = np.full((target_height, target_width, 3), PANEL_BACKGROUND, dtype=np.uint8)
    if frame is None or frame.size == 0:
        return canvas

    height, width = frame.shape[:2]
    scale = min(target_width / max(1, width), target_height / max(1, height))
    resized_width = max(1, int(round(width * scale)))
    resized_height = max(1, int(round(height * scale)))
    resized = cv2.resize(frame, (resized_width, resized_height), interpolation=cv2.INTER_AREA)

    offset_x = (target_width - resized_width) // 2
    offset_y = (target_height - resized_height) // 2
    canvas[offset_y:offset_y + resized_height, offset_x:offset_x + resized_width] = resized
    return canvas


def _render_placeholder(size: Tuple[int, int], title: str, message: str) -> np.ndarray:
    frame = np.full((size[1], size[0], 3), PANEL_BACKGROUND, dtype=np.uint8)
    cv2.putText(frame, title, (18, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.8, TEXT_COLOR, 2)
    cv2.putText(frame, message, (18, 74), cv2.FONT_HERSHEY_SIMPLEX, 0.6, TEXT_COLOR, 1)
    return frame


def render_panel(
    frame: Optional[np.ndarray],
    title: str,
    subtitle: str = "",
    highlight: bool = False,
    tile_size: Tuple[int, int] = DEFAULT_TILE_SIZE,
) -> np.ndarray:
    panel = _fit_with_padding(frame, tile_size) if frame is not None else _render_placeholder(
        tile_size,
        title,
        subtitle or "No signal",
    )

    cv2.rectangle(panel, (0, 0), (tile_size[0] - 1, tile_size[1] - 1), FRAME_COLOR, 2)

    cv2.rectangle(panel, (0, 0), (tile_size[0], 46), (0, 0, 0), cv2.FILLED)
    cv2.putText(panel, title, (12, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, TEXT_COLOR, 2)
    if subtitle:
        cv2.putText(panel, subtitle, (12, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, TEXT_COLOR, 1)
    return panel


def compose_multiview(
    feeds: Iterable[Dict[str, object]],
    fullscreen_camera_id: Optional[int] = None,
    tile_size: Tuple[int, int] = DEFAULT_TILE_SIZE,
    gap: int = 10,
) -> np.ndarray:
    feed_list = list(feeds)
    if not feed_list:
        return _render_placeholder(tile_size, "Playback", "No camera feeds available")

    visible_feeds = feed_list
    if fullscreen_camera_id is not None:
        visible_feeds = [feed for feed in feed_list if feed.get("camera_id") == fullscreen_camera_id]
        if not visible_feeds:
            visible_feeds = feed_list

    count = len(visible_feeds)
    columns = 1 if fullscreen_camera_id is not None else max(1, int(math.ceil(math.sqrt(count))))
    rows = int(math.ceil(count / columns))

    canvas_height = rows * tile_size[1] + gap * (rows + 1)
    canvas_width = columns * tile_size[0] + gap * (columns + 1)
    canvas = np.full((canvas_height, canvas_width, 3), GRID_BACKGROUND, dtype=np.uint8)

    for index, feed in enumerate(visible_feeds):
        row = index // columns
        column = index % columns
        x = gap + column * (tile_size[0] + gap)
        y = gap + row * (tile_size[1] + gap)

        panel = render_panel(
            frame=feed.get("frame"),
            title=str(feed.get("title", f"Camera {feed.get('camera_id', '?')}")),
            subtitle=str(feed.get("subtitle", "")),
            highlight=bool(feed.get("highlight")),
            tile_size=tile_size,
        )
        canvas[y:y + tile_size[1], x:x + tile_size[0]] = panel

    return canvas
