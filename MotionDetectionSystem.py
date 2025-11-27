import cv2
import time
import csv
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision

# ==========================================================
# CONFIG
# ==========================================================
CAMERA_INDEX   = 0
RECORD_TO_CSV  = True
CSV_FILENAME   = "multi_person_pose_hands_mocap.csv"

MODEL_PATH = "/Users/kanishkaroy/PycharmProjects/testing/pose_landmarker_full.task"
MAX_PEOPLE     = 10
NUM_POSE_LM    = 33
VALS_PER_LM    = 4   # x,y,z,visibility

# ---- Hands config (classic MediaPipe Hands) ----
MAX_HANDS        = 4
HAND_LM_COUNT    = 21
HAND_VALS_PER_LM = 3   # x,y,z

# ==========================================================
# MEDIAPIPE TASKS SETUP (PoseLandmarker, multi-person)
# ==========================================================
BaseOptions            = mp_tasks.BaseOptions
PoseLandmarker         = vision.PoseLandmarker
PoseLandmarkerOptions  = vision.PoseLandmarkerOptions
RunningMode            = vision.RunningMode
ImageFormat            = mp.ImageFormat
mp_image               = mp.Image

base_options = BaseOptions(model_asset_path=MODEL_PATH)

options = PoseLandmarkerOptions(
    base_options=base_options,
    running_mode=RunningMode.VIDEO,
    num_poses=MAX_PEOPLE,              # allow up to MAX_PEOPLE
    min_pose_detection_confidence=0.3, # slightly lower thresholds
    min_pose_presence_confidence=0.3,
    min_tracking_confidence=0.3,
)

landmarker = PoseLandmarker.create_from_options(options)

# ==========================================================
# MEDIAPIPE Hands (classic API)
# ==========================================================
mp_drawing = mp.solutions.drawing_utils
mp_hands   = mp.solutions.hands

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=MAX_HANDS,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)

# ==========================================================
# CSV SETUP
# ==========================================================
csv_file   = None
csv_writer = None
frame_idx  = 0
recording  = False

POSE_VALUES_TOTAL  = MAX_PEOPLE * NUM_POSE_LM * VALS_PER_LM
HAND_VALUES_TOTAL  = MAX_HANDS * HAND_LM_COUNT * HAND_VALS_PER_LM
TOTAL_VALUES       = POSE_VALUES_TOTAL + HAND_VALUES_TOTAL

if RECORD_TO_CSV:
    csv_file = open(CSV_FILENAME, "w", newline="")
    csv_writer = csv.writer(csv_file)

    header = ["frame"]
    for p in range(MAX_PEOPLE):
        for i in range(NUM_POSE_LM):
            header += [f"p{p}_lm{i}_x", f"p{p}_lm{i}_y", f"p{p}_lm{i}_z", f"p{p}_lm{i}_vis"]
    for h_idx in range(MAX_HANDS):
        for j in range(HAND_LM_COUNT):
            header += [f"h{h_idx}_j{j}_x", f"h{h_idx}_j{j}_y", f"h{h_idx}_j{j}_z"]
    csv_writer.writerow(header)
    print(f"[INFO] Recording multi-person pose + hands data to: {CSV_FILENAME}")

print("[INFO] Press 'r' to start/stop recording, 'q' to quit.")

# ==========================================================
# VIDEO CAPTURE
# ==========================================================
cap = cv2.VideoCapture(CAMERA_INDEX)
if not cap.isOpened():
    raise RuntimeError(f"Could not open camera index {CAMERA_INDEX}")

prev_time = time.time()

COLORS = [
    (0, 255, 0),
    (0, 255, 255),
    (255, 255, 0),
    (255, 0, 0),
    (255, 0, 255),
    (0, 128, 255),
    (255, 128, 0),
    (0, 200, 0),
    (200, 0, 200),
    (128, 255, 128),
]

POSE_CONNECTIONS = mp.solutions.pose.POSE_CONNECTIONS

# ==========================================================
# MAIN LOOP
# ==========================================================
try:
    while True:
        ok, frame = cap.read()
        if not ok:
            print("[WARN] Frame read failed.")
            break

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # --- PoseLandmarker (multi-person body) ---
        ts_ms = int(time.time() * 1000)
        mp_img = mp_image(image_format=ImageFormat.SRGB, data=rgb)
        result = landmarker.detect_for_video(mp_img, ts_ms)

        num_people = len(result.pose_landmarks)
        num_people_clamped = min(num_people, MAX_PEOPLE)

        # simple debug print in terminal
        print(f"\rDetected people this frame: {num_people}", end="")

        # --- Hands (classic MP Hands) ---
        hands_result = hands.process(rgb)
        num_hands_detected = 0 if not hands_result.multi_hand_landmarks else len(hands_result.multi_hand_landmarks)
        num_hands_clamped = min(num_hands_detected, MAX_HANDS)

        # --------------------------------------------------
        # DRAW ALL BODIES + ID LABELS
        # --------------------------------------------------
        for p_idx in range(num_people_clamped):
            color = COLORS[p_idx % len(COLORS)]
            landmarks = result.pose_landmarks[p_idx]

            # draw joints
            for lm in landmarks:
                cx = int(lm.x * w)
                cy = int(lm.y * h)
                cv2.circle(frame, (cx, cy), 5, color, -1)

            # draw bones
            for c in POSE_CONNECTIONS:
                i, j = c
                li = landmarks[i]
                lj = landmarks[j]
                x1, y1 = int(li.x * w), int(li.y * h)
                x2, y2 = int(lj.x * w), int(lj.y * h)
                cv2.line(frame, (x1, y1), (x2, y2), color, 3)

            # label each person at their nose
            nose = landmarks[0]  # landmark 0 = nose
            nx, ny = int(nose.x * w), int(nose.y * h)
            cv2.putText(
                frame,
                f"P{p_idx}",
                (nx + 10, ny - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2,
                cv2.LINE_AA,
            )

        # --------------------------------------------------
        # DRAW HANDS
        # --------------------------------------------------
        if hands_result.multi_hand_landmarks:
            for h_idx, hand_lms in enumerate(hands_result.multi_hand_landmarks[:MAX_HANDS]):
                hand_color = COLORS[(h_idx + 3) % len(COLORS)]
                mp_drawing.draw_landmarks(
                    frame,
                    hand_lms,
                    mp_hands.HAND_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing.DrawingSpec(
                        color=hand_color, thickness=2, circle_radius=3
                    ),
                    connection_drawing_spec=mp_drawing.DrawingSpec(
                        color=hand_color, thickness=2, circle_radius=3
                    ),
                )

        pose_info = f"People: {num_people} | Hands: {num_hands_detected}"

        cv2.putText(
            frame,
            pose_info,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0) if num_people > 0 or num_hands_detected > 0 else (0, 0, 255),
            2,
            cv2.LINE_AA,
        )

        now = time.time()
        fps = 1.0 / (now - prev_time) if now != prev_time else 0.0
        prev_time = now
        cv2.putText(
            frame,
            f"FPS: {fps:.1f}",
            (10, h - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        if recording:
            cv2.circle(frame, (w - 30, 30), 12, (0, 0, 255), -1)
            cv2.putText(
                frame,
                "REC",
                (w - 80, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )

        cv2.imshow("Multi-person Pose + Hands MoCap", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("r"):
            recording = not recording
            print(f"\n[INFO] Recording: {recording}")

        # --------------------------------------------------
        # CSV RECORDING (unchanged)
        # --------------------------------------------------
        if recording and csv_writer is not None:
            flat = [np.nan] * TOTAL_VALUES
            offset = 0

            # Pose
            for p_idx in range(MAX_PEOPLE):
                if p_idx < num_people:
                    landmarks = result.pose_landmarks[p_idx]
                    for i, lm in enumerate(landmarks):
                        if i >= NUM_POSE_LM:
                            break
                        base = offset + i * VALS_PER_LM
                        flat[base + 0] = float(lm.x)
                        flat[base + 1] = float(lm.y)
                        flat[base + 2] = float(lm.z)
                        flat[base + 3] = float(lm.visibility)
                offset += NUM_POSE_LM * VALS_PER_LM

            # Hands
            if hands_result.multi_hand_landmarks:
                for h_idx, hand_lms in enumerate(hands_result.multi_hand_landmarks[:MAX_HANDS]):
                    for j, lm in enumerate(hand_lms.landmark):
                        if j >= HAND_LM_COUNT:
                            break
                        base = POSE_VALUES_TOTAL + h_idx * HAND_LM_COUNT * HAND_VALS_PER_LM + j * HAND_VALS_PER_LM
                        flat[base + 0] = float(lm.x)
                        flat[base + 1] = float(lm.y)
                        flat[base + 2] = float(lm.z)

            row = [frame_idx] + flat
            csv_writer.writerow(row)

        frame_idx += 1

finally:
    cap.release()
    cv2.destroyAllWindows()
    landmarker.close()
    hands.close()
    if csv_file is not None:
        csv_file.close()
    print("\n[INFO] Finished.")
