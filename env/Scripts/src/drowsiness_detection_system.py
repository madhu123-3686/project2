
#!/usr/bin/env python3
"""
Real-Time Driver Drowsiness Detection System
Author: AI Research Assistant
Date: October 2025

This system uses computer vision techniques to detect driver drowsiness in real-time
by monitoring eye aspect ratio (EAR) and triggering alerts when drowsiness is detected.

Key Features:
- Real-time face and eye detection using dlib
- Eye Aspect Ratio (EAR) calculation and monitoring
- Configurable thresholds and frame counting
- Audio and visual alerts
- Multi-threading for smooth operation
- Comprehensive logging and data collection

Dependencies:
- OpenCV (cv2)
- dlib
- imutils
- scipy
- numpy
- pygame (for audio alerts)
- threading
"""

import cv2
import dlib
import numpy as np
import imutils
import time
import threading
import argparse
from scipy.spatial import distance as dist
from imutils import face_utils
from imutils.video import VideoStream
import pygame
import os
import json
from datetime import datetime

class DrowsinessDetector:
    def __init__(self, shape_predictor_path, ear_threshold=0.3, ear_consec_frames=48, 
                 alarm_path=None, webcam_index=0, frame_width=450):
        """
        Initialize the Drowsiness Detection System
        
        Args:
            shape_predictor_path (str): Path to dlib's facial landmark predictor
            ear_threshold (float): EAR threshold for drowsiness detection
            ear_consec_frames (int): Consecutive frames threshold for alarm
            alarm_path (str): Path to alarm sound file
            webcam_index (int): Webcam index for video capture
            frame_width (int): Width to resize frames for processing
        """
        
        # Configuration parameters
        self.EAR_THRESHOLD = ear_threshold
        self.EAR_CONSEC_FRAMES = ear_consec_frames
        self.FRAME_WIDTH = frame_width
        self.alarm_path = alarm_path
        
        # System state variables
        self.COUNTER = 0
        self.ALARM_ON = False
        self.total_blinks = 0
        self.session_start_time = datetime.now()
        
        # Initialize face detector and landmark predictor
        print("[INFO] Loading facial landmark predictor...")
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(shape_predictor_path)
        
        # Get eye landmark indices
        (self.lStart, self.lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
        (self.rStart, self.rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
        
        # Initialize video stream
        print("[INFO] Starting video stream...")
        self.vs = VideoStream(src=webcam_index).start()
        time.sleep(2.0)  # Allow camera to warm up
        
        # Initialize pygame for audio alerts
        if self.alarm_path and os.path.exists(self.alarm_path):
            pygame.mixer.init()
            self.alarm_sound = pygame.mixer.Sound(self.alarm_path)
        else:
            self.alarm_sound = None
            print("[WARNING] No valid alarm file provided - using visual alerts only")
        
        # Data logging
        self.session_data = []
        
        print("[INFO] Drowsiness detector initialized successfully!")
        print(f"[INFO] EAR Threshold: {self.EAR_THRESHOLD}")
        print(f"[INFO] Consecutive frames threshold: {self.EAR_CONSEC_FRAMES}")
    
    def calculate_ear(self, eye):
        """
        Calculate Eye Aspect Ratio (EAR) for given eye landmarks
        
        Args:
            eye (numpy.ndarray): Array of (x,y) coordinates for eye landmarks
            
        Returns:
            float: Eye aspect ratio value
        """
        # Compute euclidean distances between vertical eye landmarks
        A = dist.euclidean(eye[1], eye[5])
        B = dist.euclidean(eye[2], eye[4])
        
        # Compute euclidean distance between horizontal eye landmarks  
        C = dist.euclidean(eye[0], eye[3])
        
        # Calculate eye aspect ratio
        ear = (A + B) / (2.0 * C)
        return ear
    
    def sound_alarm(self):
        """Play alarm sound in separate thread"""
        if self.alarm_sound:
            self.alarm_sound.play()
    
    def log_detection_event(self, ear_value, alert_triggered, timestamp):
        """Log detection events for analysis"""
        event_data = {
            'timestamp': timestamp.isoformat(),
            'ear_value': float(ear_value),
            'alert_triggered': bool(alert_triggered),
            'consecutive_frames': int(self.COUNTER)
        }
        self.session_data.append(event_data)
    
    def save_session_data(self):
        """Save session data to JSON file"""
        session_summary = {
            'session_start': self.session_start_time.isoformat(),
            'session_end': datetime.now().isoformat(),
            'total_alerts': sum(1 for event in self.session_data if event['alert_triggered']),
            'total_frames_processed': len(self.session_data),
            'detection_events': self.session_data
        }
        
        filename = f"drowsiness_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(filename, 'w') as f:
            json.dump(session_summary, f, indent=2)
        
        print(f"[INFO] Session data saved to {filename}")
        print(f"[INFO] Total alerts triggered: {session_summary['total_alerts']}")
        print(f"[INFO] Total frames processed: {session_summary['total_frames_processed']}")
    
    def detect_drowsiness(self):
        """Main detection loop"""
        print("[INFO] Starting drowsiness detection...")
        print("[INFO] Press 'q' to quit")
        
        frame_count = 0
        
        try:
            while True:
                # Read frame from video stream
                frame = self.vs.read()
                if frame is None:
                    break
                
                frame_count += 1
                current_time = datetime.now()
                
                # Resize and convert to grayscale
                frame = imutils.resize(frame, width=self.FRAME_WIDTH)
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                # Detect faces
                rects = self.detector(gray, 0)
                
                # Process each detected face
                for rect in rects:
                    # Get facial landmarks
                    shape = self.predictor(gray, rect)
                    shape = face_utils.shape_to_np(shape)
                    
                    # Extract left and right eye coordinates
                    leftEye = shape[self.lStart:self.lEnd]
                    rightEye = shape[self.rStart:self.rEnd]
                    
                    # Calculate EAR for both eyes
                    leftEAR = self.calculate_ear(leftEye)
                    rightEAR = self.calculate_ear(rightEye)
                    
                    # Average EAR for both eyes
                    ear = (leftEAR + rightEAR) / 2.0
                    
                    # Visualize eye regions
                    leftEyeHull = cv2.convexHull(leftEye)
                    rightEyeHull = cv2.convexHull(rightEye)
                    cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
                    cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
                    
                    # Check for drowsiness
                    alert_triggered = False
                    if ear < self.EAR_THRESHOLD:
                        self.COUNTER += 1
                        
                        # If eyes closed for sufficient consecutive frames
                        if self.COUNTER >= self.EAR_CONSEC_FRAMES:
                            if not self.ALARM_ON:
                                self.ALARM_ON = True
                                alert_triggered = True
                                
                                # Play alarm in separate thread
                                if self.alarm_sound:
                                    alarm_thread = threading.Thread(target=self.sound_alarm)
                                    alarm_thread.daemon = True
                                    alarm_thread.start()
                            
                            # Draw alert on frame
                            cv2.putText(frame, "DROWSINESS ALERT!", (10, 30),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    else:
                        self.COUNTER = 0
                        self.ALARM_ON = False
                    
                    # Log detection event
                    self.log_detection_event(ear, alert_triggered, current_time)
                    
                    # Display EAR value
                    cv2.putText(frame, f"EAR: {ear:.3f}", (300, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    
                    # Display counter
                    cv2.putText(frame, f"Frame Count: {self.COUNTER}", (10, 60),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                
                # Display status information
                status_color = (0, 255, 0) if not self.ALARM_ON else (0, 0, 255)
                status_text = "AWAKE" if not self.ALARM_ON else "DROWSY"
                cv2.putText(frame, f"Status: {status_text}", (10, frame.shape[0] - 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
                
                # Show frame
                cv2.imshow("Drowsiness Detection", frame)
                
                # Check for quit key
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                
                # Process frames at reasonable rate
                time.sleep(0.03)  # ~30 FPS
                
        except KeyboardInterrupt:
            print("[INFO] Detection interrupted by user")
        
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Clean up resources"""
        print("[INFO] Cleaning up...")
        self.save_session_data()
        cv2.destroyAllWindows()
        self.vs.stop()
        if pygame.mixer.get_init():
            pygame.mixer.quit()

def main():
    """Main function to run the drowsiness detection system"""
    parser = argparse.ArgumentParser(description="Real-time Drowsiness Detection System")
    parser.add_argument("-p", "--shape-predictor", required=True,
                       help="path to facial landmark predictor")
    parser.add_argument("-a", "--alarm", type=str, default="",
                       help="path to alarm .WAV file")
    parser.add_argument("-w", "--webcam", type=int, default=0,
                       help="index of webcam on system")
    parser.add_argument("-t", "--threshold", type=float, default=0.3,
                       help="EAR threshold for drowsiness detection")
    parser.add_argument("-f", "--frames", type=int, default=48,
                       help="consecutive frames threshold for alarm")
    
    args = parser.parse_args()
    
    # Validate shape predictor file
    if not os.path.exists(args.shape_predictor):
        print(f"[ERROR] Shape predictor file not found: {args.shape_predictor}")
        print("[INFO] Download from: http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2")
        return
    
    # Initialize and run detector
    try:
        detector = DrowsinessDetector(
            shape_predictor_path=args.shape_predictor,
            ear_threshold=args.threshold,
            ear_consec_frames=args.frames,
            alarm_path=args.alarm if args.alarm else None,
            webcam_index=args.webcam
        )
        
        detector.detect_drowsiness()
        
    except Exception as e:
        print(f"[ERROR] Failed to initialize drowsiness detector: {e}")
        return

if __name__ == "__main__":
    main()
