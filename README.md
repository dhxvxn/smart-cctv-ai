# 🎥 AI Smart CCTV System

An AI-powered surveillance system that allows you to **search CCTV footage using natural language** instead of manually watching hours of video.

---

## 🚀 Features

- 🔍 Natural language search  
  → "cars entering zone", "person near gate"

- 🧠 Object Detection  
  → YOLOv8

- 🎯 Multi-object Tracking  
  → DeepSORT / ByteTrack

- 📍 Zone-based Event Detection  
  → Detect entry/exit in restricted areas

- 🗂️ Event Logging  
  → Stores track_id, timestamps, object type

- ⚡ Fast Video Retrieval  
  → Metadata-driven playback (NO re-tracking)

- 🎯 Accurate Highlighting  
  → Highlights exact object using track_id

---

## 🧠 Modes

### 1. Full Surveillance Mode
- Detects all objects
- Tracks movement
- Logs events when objects enter/leave zones

---

### 2. Query Mode
- Ask queries like:
  - "cars entering zone"
  - "person near parking"
- Shows filtered events
- Retrieve video instantly

---

## ⚡ Key Innovation

### 🔥 Metadata-Driven Playback

- No re-running tracker
- Instant video seek
- Accurate object highlighting
- No ID mismatch

---

## 🛠 Tech Stack

- Python
- OpenCV
- YOLOv8 (Ultralytics)
- DeepSORT / ByteTrack
- SQLite

---

## 📁 Project Structure

- detection + tracking
- event logging
- query system
- video playback engine

---

## 📌 Future Improvements

- Multi-camera tracking (ReID)
- Real-time alerts
- Web dashboard

---

## 👨‍💻 Author

Dhavan (CSE Student)

---

## ⭐ Note

This project demonstrates how AI can make CCTV systems **searchable, intelligent, and efficient**.
