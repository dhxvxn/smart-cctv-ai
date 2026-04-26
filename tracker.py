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

        print("\n================ FRAME START ================")
        print(f"[INFO] Total detections: {len(detections)}")

        ds_detections = []

        for i, (x1, y1, x2, y2, conf, cls_id) in enumerate(detections):
            w = x2 - x1
            h = y2 - y1

            print(f"[DETECTION {i}] "
                  f"BBOX=({x1},{y1},{x2},{y2}) "
                  f"CONF={conf:.2f} CLASS={cls_id}")

            ds_detections.append((
                [x1, y1, w, h],
                conf,
                cls_id
            ))

        # Update tracker
        tracks = self.tracker.update_tracks(ds_detections, frame=frame)

        print(f"[INFO] Tracks returned: {len(tracks)}")

        results = []

        for track in tracks:
            if not track.is_confirmed():
                print(f"[TRACK {track.track_id}] Not confirmed → skipped")
                continue

            l, t, r, b = track.to_ltrb()

            print(f"""
[TRACK CONFIRMED]
Track ID: {track.track_id}
BBOX: ({int(l)}, {int(t)}, {int(r)}, {int(b)})
Class: {track.det_class}
""")

            results.append((
                int(l),
                int(t),
                int(r),
                int(b),
                int(track.track_id),
                track.det_class
            ))

        print("[INFO] Final tracked objects:", len(results))
        print("================ FRAME END ==================\n")

        return results