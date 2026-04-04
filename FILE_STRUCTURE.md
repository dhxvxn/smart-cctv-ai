# Project Structure

smart_cctv/
│
├── app.py                  # Main entry point
├── config.py               # Configurations
│
├── detection/
│   └── yolo_detector.py
│
├── tracking/
│   └── tracker.py
│
├── zones/
│   ├── zone_drawer.py
│   └── zone_manager.py
│
├── events/
│   ├── event_detector.py
│   └── logger.py
│
├── database/
│   └── db.py
│
├── query/
│   └── query_parser.py
│
├── retrieval/
│   └── fetch_events.py
│
├── video/
│   ├── clip_extractor.py
│   └── player.py
│
├── ui/
│   └── console_ui.py
│
├── search_console.py       # Query interface
├── zones.json
└── events.db