from deep_sort_realtime.deepsort_tracker import DeepSort


class PersonTracker:
    def __init__(self):
        self.tracker = DeepSort(
            max_age=50,
            n_init=3,
            max_cosine_distance=0.3
        )

    def update(self, frame, detections):
        """
        detections format:
        [(x1, y1, x2, y2, conf, cls_id)]
        """

        ds_detections = []

        for (x1, y1, x2, y2, conf, cls_id) in detections:
            w = x2 - x1
            h = y2 - y1

            ds_detections.append((
                [x1, y1, w, h],
                conf,
                cls_id
            ))

        tracks = self.tracker.update_tracks(ds_detections, frame=frame)

        results = []

        for track in tracks:
            if not track.is_confirmed():
                continue

            l, t, r, b = track.to_ltrb()

            results.append((
                int(l),
                int(t),
                int(r),
                int(b),
                int(track.track_id),
                track.det_class
            ))

        return results