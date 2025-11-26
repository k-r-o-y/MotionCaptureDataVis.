import cv2
import time
import csv
import numpy as np
import mediapipe as mp

# -----------------------------------------------------------
# Configuration
# -----------------------------------------------------------
CAMERA_INDEX   = 0
RECORD_TO_CSV  = True
CSV_FILENAME   = "full_body_extended_mocap.csv"

# -----------------------------------------------------------
# MediaPipe setup (Pose + Hands + FaceMesh)
# -----------------------------------------------------------
mp_pose      = mp.solutions.pose
mp_drawing   = mp.solutions.drawing_utils
mp_hands     = mp.solutions.hands
mp_face_mesh = mp.solutions.face_mesh

pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    enable_segmentation=False,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6,
)

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)

face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)

# -----------------------------------------------------------
# Landmark / sensor layout
# -----------------------------------------------------------
NUM_POSE_LM        = 33
POSE_VALUES_PER_LM = 4   # x, y, z, visibility

NUM_VIRTUAL_JOINTS     = 3   # neck_center, pelvis_center, spine_mid
VIRT_VALUES_PER_JOINT  = 4   # x, y, z, visibility

NUM_HANDS_MAX      = 2
HAND_LM_PER_HAND   = 21
HAND_VALUES_PER_LM = 3   # x, y, z

# FaceMesh key landmark indices (MediaPipe FaceMesh)
FACE_KEYPOINTS = {
    "nose_tip":        1,
    "left_eye_outer":  33,
    "right_eye_outer": 263,
    "mouth_left":      61,
    "mouth_right":     291,
    "chin":            199,
}
FACE_VALUES_PER_LM = 3   # x, y, z

# -----------------------------------------------------------
# Video capture setup
# -----------------------------------------------------------
cap = cv2.VideoCapture(CAMERA_INDEX)
if not cap.isOpened():
    raise RuntimeError(f"Could not open camera index {CAMERA_INDEX}")

# -----------------------------------------------------------
# CSV setup
# -----------------------------------------------------------
csv_file   = None
csv_writer = None
frame_idx  = 0
recording  = False

pose_total_vals   = NUM_POSE_LM * POSE_VALUES_PER_LM
virt_total_vals   = NUM_VIRTUAL_JOINTS * VIRT_VALUES_PER_JOINT
hands_total_vals  = NUM_HANDS_MAX * HAND_LM_PER_HAND * HAND_VALUES_PER_LM
face_total_vals   = len(FACE_KEYPOINTS) * FACE_VALUES_PER_LM

TOTAL_VALUES = pose_total_vals + virt_total_vals + hands_total_vals + face_total_vals

if RECORD_TO_CSV:
    csv_file = open(CSV_FILENAME, "w", newline="")
    csv_writer = csv.writer(csv_file)

    header = ["frame"]

    # Pose landmarks
    for i in range(NUM_POSE_LM):
        header += [
            f"pose_lm{i}_x",
            f"pose_lm{i}_y",
            f"pose_lm{i}_z",
            f"pose_lm{i}_vis",
        ]

    # Virtual joints
    virt_names = ["neck_center", "pelvis_center", "spine_mid"]
    for name in virt_names:
        header += [f"{name}_x", f"{name}_y", f"{name}_z", f"{name}_vis"]

    # Hands
    for hand_tag in ["hand0", "hand1"]:
        for j in range(HAND_LM_PER_HAND):
            header += [
                f"{hand_tag}_j{j}_x",
                f"{hand_tag}_j{j}_y",
                f"{hand_tag}_j{j}_z",
            ]

    # Face keypoints
    for name in FACE_KEYPOINTS.keys():
        header += [f"face_{name}_x", f"face_{name}_y", f"face_{name}_z"]

    csv_writer.writerow(header)
    print(f"[INFO] Recording extended mocap data to: {CSV_FILENAME}")

print("[INFO] Press 'r' to start/stop recording, 'q' to quit.")

prev_time = time.time()

# -----------------------------------------------------------
# Helper: midpoint of two pose landmarks
# -----------------------------------------------------------
def midpoint_with_vis(lm_a, lm_b):
    """Return (x,y,z,vis) midpoint of two pose landmarks."""
    x = (lm_a.x + lm_b.x) / 2.0
    y = (lm_a.y + lm_b.y) / 2.0
    z = (lm_a.z + lm_b.z) / 2.0
    vis = min(lm_a.visibility, lm_b.visibility)
    return x, y, z, vis

# -----------------------------------------------------------
# Main loop
# -----------------------------------------------------------
try:
    while True:
        ok, frame = cap.read()
        if not ok:
            print("[WARN] Frame read failed.")
            break

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        pose_result  = pose.process(rgb)
        hands_result = hands.process(rgb)
        face_result  = face_mesh.process(rgb)

        # ---------------------------------------------------
        # Draw pose skeleton + spine line & virtual joints
        # ---------------------------------------------------
        neck_center   = None  # normalized coords
        pelvis_center = None
        spine_mid     = None

        if pose_result.pose_landmarks:
            lms = pose_result.pose_landmarks.landmark

            # Draw base pose with thicker lines
            mp_drawing.draw_landmarks(
                frame,
                pose_result.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing.DrawingSpec(
                    color=(0, 255, 0),
                    thickness=4,          # thicker joint markers
                    circle_radius=4,
                ),
                connection_drawing_spec=mp_drawing.DrawingSpec(
                    color=(0, 150, 255),
                    thickness=5,          # thicker skeleton lines
                    circle_radius=4,
                ),
            )

            # --------- Compute virtual joints for drawing ---------
            L_SH = mp_pose.PoseLandmark.LEFT_SHOULDER.value
            R_SH = mp_pose.PoseLandmark.RIGHT_SHOULDER.value
            L_HIP = mp_pose.PoseLandmark.LEFT_HIP.value
            R_HIP = mp_pose.PoseLandmark.RIGHT_HIP.value

            neck_x, neck_y, neck_z, neck_vis = midpoint_with_vis(lms[L_SH], lms[R_SH])
            pelvis_x, pelvis_y, pelvis_z, pelvis_vis = midpoint_with_vis(lms[L_HIP], lms[R_HIP])

            spine_x = (neck_x + pelvis_x) / 2.0
            spine_y = (neck_y + pelvis_y) / 2.0
            spine_z = (neck_z + pelvis_z) / 2.0
            spine_vis = min(neck_vis, pelvis_vis)

            neck_center   = (neck_x, neck_y, neck_z, neck_vis)
            pelvis_center = (pelvis_x, pelvis_y, pelvis_z, pelvis_vis)
            spine_mid     = (spine_x, spine_y, spine_z, spine_vis)

            # --------- Draw spine line + markers (thicker) ---------
            def to_px(p):
                return int(p[0] * w), int(p[1] * h)

            neck_px   = to_px(neck_center)
            pelvis_px = to_px(pelvis_center)
            spine_px  = to_px(spine_mid)

            # Spine line (centre of torso) – thicker
            cv2.line(
                frame,
                pelvis_px,
                neck_px,
                (255, 255, 0),
                6,   # thicker spine line
            )

            # Neck / pelvis / spine markers – bigger circles
            cv2.circle(frame, neck_px,   8, (0, 255, 255), -1)   # neck center
            cv2.circle(frame, pelvis_px, 8, (0, 200, 255), -1)   # pelvis center
            cv2.circle(frame, spine_px,  8, (255, 255, 255), -1) # spine midpoint

            pose_info = "Pose detected (with spine line)"
        else:
            pose_info = "No pose"

        # ---------------------------------------------------
        # Draw hands (thicker)
        # ---------------------------------------------------
        hand_info = ""
        if hands_result.multi_hand_landmarks:
            num_hands = min(len(hands_result.multi_hand_landmarks), NUM_HANDS_MAX)
            hand_info = f" | Hands: {num_hands}"
            for hi in range(num_hands):
                hand_lm = hands_result.multi_hand_landmarks[hi]
                mp_drawing.draw_landmarks(
                    frame,
                    hand_lm,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(
                        color=(0, 255, 255),
                        thickness=4,
                        circle_radius=4
                    ),
                    mp_drawing.DrawingSpec(
                        color=(255, 255, 0),
                        thickness=5,
                        circle_radius=4
                    ),
                )
        else:
            hand_info = " | Hands: 0"

        # ---------------------------------------------------
        # Draw face keypoints (bigger)
        # ---------------------------------------------------
        face_info = ""
        if face_result.multi_face_landmarks:
            face_lms = face_result.multi_face_landmarks[0].landmark
            face_info = " | Face: 1"
            for name, idx in FACE_KEYPOINTS.items():
                lm = face_lms[idx]
                cx = int(lm.x * w)
                cy = int(lm.y * h)
                cv2.circle(frame, (cx, cy), 5, (255, 0, 255), -1)  # bigger face dots
        else:
            face_info = " | Face: 0"

        # HUD text
        info_text = pose_info + hand_info + face_info
        cv2.putText(
            frame,
            info_text,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0) if "Pose detected" in pose_info else (0, 0, 255),
            2,
            cv2.LINE_AA
        )

        # FPS
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
            cv2.LINE_AA
        )

        # Recording indicator (can leave as is or make bigger)
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
                cv2.LINE_AA
            )

        cv2.imshow("Full Body Extended MoCap (with spine line)", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("r"):
            recording = not recording
            print(f"[INFO] Recording: {recording}")

        # ---------------------------------------------------
        # CSV recording
        # ---------------------------------------------------
        if recording and csv_writer is not None:
            flat = [np.nan] * TOTAL_VALUES
            offset = 0

            # Pose landmarks
            if pose_result.pose_landmarks:
                lms = pose_result.pose_landmarks.landmark
                for i, lm in enumerate(lms):
                    if i >= NUM_POSE_LM:
                        break
                    base = offset + i * POSE_VALUES_PER_LM
                    flat[base + 0] = float(lm.x)
                    flat[base + 1] = float(lm.y)
                    flat[base + 2] = float(lm.z)
                    flat[base + 3] = float(lm.visibility)
            offset += pose_total_vals

            # Virtual joints (neck_center, pelvis_center, spine_mid)
            if pose_result.pose_landmarks:
                lms = pose_result.pose_landmarks.landmark
                L_SH = mp_pose.PoseLandmark.LEFT_SHOULDER.value
                R_SH = mp_pose.PoseLandmark.RIGHT_SHOULDER.value
                L_HIP = mp_pose.PoseLandmark.LEFT_HIP.value
                R_HIP = mp_pose.PoseLandmark.RIGHT_HIP.value

                neck_x, neck_y, neck_z, neck_vis = midpoint_with_vis(lms[L_SH], lms[R_SH])
                pelvis_x, pelvis_y, pelvis_z, pelvis_vis = midpoint_with_vis(lms[L_HIP], lms[R_HIP])
                spine_x = (neck_x + pelvis_x) / 2.0
                spine_y = (neck_y + pelvis_y) / 2.0
                spine_z = (neck_z + pelvis_z) / 2.0
                spine_vis = min(neck_vis, pelvis_vis)

                virtuals = [
                    (neck_x,   neck_y,   neck_z,   neck_vis),
                    (pelvis_x, pelvis_y, pelvis_z, pelvis_vis),
                    (spine_x,  spine_y,  spine_z,  spine_vis),
                ]

                for j, (x, y, z, vis) in enumerate(virtuals):
                    base = offset + j * VIRT_VALUES_PER_JOINT
                    flat[base + 0] = float(x)
                    flat[base + 1] = float(y)
                    flat[base + 2] = float(z)
                    flat[base + 3] = float(vis)
            offset += virt_total_vals

            # Hands
            if hands_result.multi_hand_landmarks:
                num_hands = min(len(hands_result.multi_hand_landmarks), NUM_HANDS_MAX)
                for hi in range(num_hands):
                    hand_lm = hands_result.multi_hand_landmarks[hi]
                    for ji, lm in enumerate(hand_lm.landmark):
                        if ji >= HAND_LM_PER_HAND:
                            break
                        base = offset + hi * HAND_LM_PER_HAND * HAND_VALUES_PER_LM + ji * HAND_VALUES_PER_LM
                        flat[base + 0] = float(lm.x)
                        flat[base + 1] = float(lm.y)
                        flat[base + 2] = float(lm.z)
            offset += hands_total_vals

            # Face keypoints
            if face_result.multi_face_landmarks:
                face_lms = face_result.multi_face_landmarks[0].landmark
                for k, (name, idx) in enumerate(FACE_KEYPOINTS.items()):
                    lm = face_lms[idx]
                    base = offset + k * FACE_VALUES_PER_LM
                    flat[base + 0] = float(lm.x)
                    flat[base + 1] = float(lm.y)
                    flat[base + 2] = float(lm.z)

            row = [frame_idx] + flat
            csv_writer.writerow(row)

        frame_idx += 1

finally:
    cap.release()
    cv2.destroyAllWindows()
    pose.close()
    hands.close()
    face_mesh.close()
    if csv_file is not None:
        csv_file.close()
    print("[INFO] Finished.")
