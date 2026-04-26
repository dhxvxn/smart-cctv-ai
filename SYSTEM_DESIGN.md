# Smart CCTV System - Final Design

---

## OVERVIEW

This system is an AI-powered Smart CCTV solution that enables:

- Real-time human detection and tracking
- Cross-camera identity matching (Global ID)
- Event-based logging (single + multi camera)
- Query-based retrieval and playback

---

## COMPLETE PIPELINE

Camera Feed
→ Detection (YOLO)
→ Tracking (DeepSORT)
→ Feature Extraction (ReID + Color)
→ Global ID Assignment
→ Event Manager
→ Database Storage
→ Query System
→ Video Playback

---

## MODULE BREAKDOWN

---

### 1. Detection Module

- Model: YOLOv8
- Detects humans in each frame

---

### 2. Tracking Module

- Algorithm: DeepSORT
- Tracks objects within a single camera
- Outputs:
  - bounding box
  - track_id (local to camera)

---

### 3. Feature Extraction Module

For each detected person:

Extract:

- ReID embedding (FastReID)
- Shirt color (upper body region)
- Timestamp
- Camera ID

---

### 4. Global ID Assignment (CORE MODULE)

This module assigns a consistent identity across cameras.

---

#### FEATURE STRUCTURE

feature = {
    embedding,
    color,
    timestamp,
    camera_id
}

---

#### TRACK MEMORY

track_memory = {
    gid: {
        embeddings: last 10,
        colors: last 10,
        last_seen,
        camera_history
    }
}

---

#### MATCHING STRATEGY (FINAL)

##### Stage 1: Strong Match

IF cosine_similarity > 0.80:

    assign same global_id

---

##### Stage 2: Soft Matching

score =
    0.7 * reid_similarity
  + 0.05 * color_similarity
  - 0.2 * time_penalty

---

#### TIME CONSTRAINT

MAX_TIME_DIFF = 10 sec

time_penalty = min(time_diff / 10, 1.0)

---

#### CANDIDATE FILTER

Only compare with:

(now - last_seen) < 10 sec

---

#### DECISION

IF score > 0.45:

    assign same global_id

ELSE:

    create new global_id

---

#### COLOR RULE

- Use color as bonus only
- NEVER penalize based on color

---

#### MEMORY UPDATE

- Store last 10 embeddings
- Store last 10 colors
- Use average embedding for matching

---

#### RESULT

- Same person across cameras → SAME global_id
- Stable identity tracking
- Reduced ID switching

---

### 5. Event Manager (DUAL MODE)

Supports two modes:

---

#### SINGLE CAMERA MODE

- Logs events per camera independently
- Uses camera_id

---

#### MULTI CAMERA MODE

- Merges events using global_id
- Tracks movement across cameras

---

#### DATA STRUCTURE

active_events = {
    gid: {
        start_time,
        last_seen,
        cameras,
        frames
    }
}

---

#### FINALIZATION

IF now - last_seen > TIMEOUT:

    save event

---

### 6. Database Layer

Stores:

- global_id
- timestamps
- camera_id(s)
- event_mode (single / multi)

---

### 7. Query System (SESSION MODE)

---

#### START BEHAVIOR

On running search_console.py:

Ask user:

"Select mode:
1. Single Camera Events
2. Multi Camera Events"

---

#### SESSION LOCK

- Store selection as session_mode
- Do NOT ask again

---

#### QUERY FLOW

WHILE running:

- Take user input
- Filter events based on session_mode
- Return results

---

#### EXIT

IF user types "exit":

    terminate program

---

### 8. Video Playback

- Retrieves full event timeline
- Shows:
  - entry
  - movement
  - exit

---

#### DISPLAY RULE

- Highlight ONLY bounding box
- DO NOT highlight full frame

---

### 9. Debugging System

Print for each match:

- global_id candidate
- reid similarity
- color similarity
- time difference
- final score

---

#### PURPOSE

- Tune thresholds
- Identify mismatches
- Improve accuracy

---

## FINAL SYSTEM CHARACTERISTICS

- Real-time processing
- Multi-camera tracking
- Stable global identity
- Event-based retrieval
- Clean query experience

---

## IMPORTANT RULES

- REMOVE any old logic:
    - similarity > 0.75
    - time_diff < 3 sec

- Use ONLY two-stage matching system

---

## FINAL RESULT

- Same person across cameras → SAME global_id
- No event duplication
- Stable tracking in crowded scenes
- Clean user interaction flow