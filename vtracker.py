import os
import cv2
import numpy as np

class PersonTracker:
    def __init__(self, video_file):
        # Load YOLO model
        self.net = cv2.dnn.readNet("cfg/yolov4.weights", "cfg/yolov4.cfg")
        with open("cfg/coco.names", "r") as f:
            self.classes = [line.strip() for line in f.readlines()]
        layer_names = self.net.getLayerNames()
        self.output_layers = [layer_names[i - 1] for i in self.net.getUnconnectedOutLayers()]

        # Open video and Get video properties
        if not os.path.isfile(video_file):
            print("File not found.")
            exit()
        self.cap = cv2.VideoCapture(video_file)
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0  # Use 30 FPS if retrieval fails

        # Initialize tracker
        self.tracker = cv2.TrackerCSRT_create()
        self.selected_bbox = None  # Stores the manually selected bounding box
        self.tracking_active = False  # Flag to switch between YOLO detection and tracking
        cv2.namedWindow("Select Person", cv2.WINDOW_NORMAL)
        cv2.setMouseCallback("Select Person", self.select_person)

    def select_person(self, event, x, y, flags, param):
        """Handles mouse clicks for selecting a person to track."""
        if event == cv2.EVENT_LBUTTONDOWN:
            for box in self.boxes:
                bx, by, bw, bh = box
                if bx <= x <= bx + bw and by <= y <= by + bh:
                    self.selected_bbox = tuple(box)
                    self.tracker.init(self.frame, self.selected_bbox)
                    self.tracking_active = True
                    print(f"Selected Bounding Box: {self.selected_bbox}")
                    break

    def detect_person(self):
        """Detects persons using YOLO and prompts the user to select one."""
        for out in self.outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5 and self.classes[class_id] == "person":
                    center_x = int(detection[0] * self.width)
                    center_y = int(detection[1] * self.height)
                    w = int(detection[2] * self.width)
                    h = int(detection[3] * self.height)
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    self.boxes.append([x, y, w, h])
                    self.confidences.append(float(confidence))
                    self.class_ids.append(class_id)

        indexes = cv2.dnn.NMSBoxes(self.boxes, self.confidences, 0.5, 0.4)
        if len(indexes) > 1:
            while self.selected_bbox is None:
                display_frame = self.frame.copy()
                for i in indexes:
                    x, y, w, h = self.boxes[i]
                    cv2.rectangle(display_frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                cv2.putText(display_frame, "Click on a person to track", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                # Ensure correct window scaling
                resized_frame = cv2.resize(display_frame, (self.frame_width, self.frame_height))
                cv2.imshow("Select Person", resized_frame)

                if cv2.waitKey(1) & 0xFF == 27:
                    self.cap.release()
                    cv2.destroyAllWindows()
                    exit()

            try:
                cv2.destroyWindow("Select Person")
            except cv2.error:
                pass

    def run(self):
        """Main loop for detection and tracking."""
        while self.cap.isOpened():
            ret, self.frame = self.cap.read()
            if not ret:
                break

            self.height, self.width, _ = self.frame.shape
            if not self.tracking_active:
                blob = cv2.dnn.blobFromImage(self.frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
                self.net.setInput(blob)
                self.outs = self.net.forward(self.output_layers)
                self.boxes = []
                self.confidences = []
                self.class_ids = []
                self.detect_person()
            else:
                success, self.selected_bbox = self.tracker.update(self.frame)
                if success:
                    x, y, w, h = [int(v) for v in self.selected_bbox]
                    cv2.rectangle(self.frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.putText(self.frame, "Selected Person", (x, y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                else:
                    self.tracking_active = False

            cv2.imshow("Tracking", self.frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break

    def exit(self):
        """Cleans up resources before exiting."""
        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    tracker = PersonTracker('data/vtest1.mp4')
    tracker.run()
    tracker.exit()