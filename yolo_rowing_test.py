#run pip install -r requirements.txt
import argparse
import cv2
import sys
import numpy as np
from ultralytics import YOLO

class RowingDetector:
    def __init__(self):
        self.state = "EXTENDED"  # "EXTENDED" or "PULLED"
        self.stroke_count = 0
        self.boat_speed = 0.0

    def update(self, left_wrist, right_wrist, left_shoulder, right_shoulder):
        # Calculate average wrist Y position and average shoulder Y position
        avg_wrist_y = (left_wrist[1] + right_wrist[1]) / 2.0
        avg_shoulder_y = (left_shoulder[1] + right_shoulder[1]) / 2.0
        
        # Distance ratio relative to shoulders
        wrist_dist = avg_wrist_y - avg_shoulder_y

        # Thresholds (adjust based on camera angle and distance)
        PULL_THRESHOLD = 80   # Wrists pulled down toward chest
        RESET_THRESHOLD = 30  # Wrists extended forward

        # State machine for stroke cycle
        if self.state == "EXTENDED" and wrist_dist > PULL_THRESHOLD:
            self.state = "PULLED"
            self.stroke_count += 1
            self.boat_speed += 5.0  # Apply a speed boost per stroke
            
        elif self.state == "PULLED" and wrist_dist < RESET_THRESHOLD:
            self.state = "EXTENDED"

        # Natural water drag (decay speed over time)
        self.boat_speed = max(0.0, self.boat_speed - 0.1)
        return self.stroke_count, self.boat_speed

def main(input_source):
    print("Loading YOLOv8 Pose model...")
    model = YOLO('yolov8n-pose.pt')

    cap = cv2.VideoCapture(int(input_source) if input_source.isdigit() else input_source)
    if not cap.isOpened():
        sys.exit("Error: Could not open webcam.")

    # Dictionary to maintain separate stroke detectors for each player ID
    player_detectors = {}

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        results = model.track(source=frame, persist=True, show=False, verbose=False)

        for result in results:
            if result.boxes.id is not None and result.keypoints is not None:
                track_ids = result.boxes.id.int().cpu().tolist()
                keypoints = result.keypoints.xy.cpu().numpy()

                for i, track_id in enumerate(track_ids):
                    if track_id not in player_detectors:
                        player_detectors[track_id] = RowingDetector()

                    # Extract Keypoints (5: L_Shoulder, 6: R_Shoulder, 9: L_Wrist, 10: R_Wrist)
                    l_sh = keypoints[i][5]
                    r_sh = keypoints[i][6]
                    l_wr = keypoints[i][9]
                    r_wr = keypoints[i][10]

                    # Verify keypoints are detected
                    if all(pt[0] != 0 for pt in [l_sh, r_sh, l_wr, r_wr]):
                        strokes, speed = player_detectors[track_id].update(l_wr, r_wr, l_sh, r_sh)

                        # Draw feedback on screen
                        px, py = int(l_sh[0]), int(l_sh[1]) - 40
                        cv2.putText(frame, f"P{track_id} Strokes: {strokes} | Speed: {speed:.1f} kt", 
                                    (px, py), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        annotated_frame = results[0].plot()
        cv2.imshow("Rowing Game - Motion Input Test", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='YOLOv8 Rowing Control Test')
    parser.add_argument('-i', '--input', default='0', help='Camera index or video file')
    args = parser.parse_args()
    main(args.input)
