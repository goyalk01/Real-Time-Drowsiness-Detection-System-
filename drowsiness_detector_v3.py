import cv2
import dlib
import numpy as np
import tensorflow as tf
from keras.models import load_model
import time
from collections import deque
import pygame
import threading
import queue
import os
import sys
import atexit
import argparse

# === GPU Configuration ===
# Force TensorFlow to see GPU and use it properly
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# Configure TensorFlow GPU
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Enable memory growth
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        # Set GPU as default
        tf.config.set_visible_devices(gpus[0], 'GPU')
        print(f"✅ GPU acceleration enabled - RTX 4050 detected and configured")
        GPU_AVAILABLE = True
    except RuntimeError as e:
        print(f"⚠️ GPU setup error: {e}")
        GPU_AVAILABLE = False
else:
    print("⚠️ GPU not found - using CPU")
    GPU_AVAILABLE = False

# === Configuration ===
MODEL_PATH = r"C:\Users\divya\DDD_CNN_PE\drowsiness_cnn_final_1_model.h5"
PREDICTOR_PATH = r"C:\Users\divya\DDD_CNN_PE\shape_predictor_68_face_landmarks.dat"

# Optimized parameters - no frame skipping for smooth visuals
CNN_PREDICTION_INTERVAL = 2      # CNN every 2 frames (smooth but efficient)
FACE_DETECTION_SCALE = 0.75      # Less aggressive scaling for better accuracy

# Detection parameters
DROWSINESS_THRESHOLD = 0.55
EYE_AR_THRESHOLD = 0.25
EYE_AR_CONSEC_FRAMES = 12        # Slightly reduced for faster response
CONSECUTIVE_FRAMES = 3           # Increased to reduce false alerts
SMOOTHING_WINDOW = 5             # Increased for smoother readings
ALERT_COOLDOWN = 3               # Increased to reduce spam
FACE_LOST_THRESHOLD = 30
MANUAL_CONFIG_DURATION = 8

# Eye landmark indices
LEFT_EYE = [36, 37, 38, 39, 40, 41]
RIGHT_EYE = [42, 43, 44, 45, 46, 47]

# Color scheme
COLORS = {
    'green': (0, 255, 0),
    'red': (0, 0, 255),
    'yellow': (0, 255, 255),
    'blue': (255, 0, 0),
    'white': (255, 255, 255),
    'gray': (128, 128, 128)
}

# Global variables
cap = None
audio_manager = None

def calculate_eye_aspect_ratio(eye_landmarks):
    """Optimized EAR calculation"""
    A = np.linalg.norm(eye_landmarks[1] - eye_landmarks[5])
    B = np.linalg.norm(eye_landmarks[2] - eye_landmarks[4])
    C = np.linalg.norm(eye_landmarks[0] - eye_landmarks[3])
    return (A + B) / (2.0 * C)

class AudioManager:
    """Professional audio manager with enhanced sounds"""
    
    def __init__(self):
        self.enabled = False
        self.sound_queue = queue.Queue(maxsize=5)
        self.shutdown = False
        
        try:
            pygame.mixer.pre_init(frequency=22050, size=-16, channels=2, buffer=512)
            pygame.mixer.init()
            self.enabled = True
            self._create_enhanced_sounds()
            self.worker_thread = threading.Thread(target=self._sound_worker, daemon=True)
            self.worker_thread.start()
            print("🔊 Enhanced audio system initialized")
        except Exception as e:
            print(f"🔇 Audio disabled: {e}")
    
    def _create_enhanced_sounds(self):
        """Create professional alert sounds"""
        sample_rate = 22050
        
        # Drowsiness alert - urgent 3-tone sequence
        duration = 1.5
        frames = int(duration * sample_rate)
        drowsy_arr = np.zeros(frames)
        
        # Create urgent alarm pattern
        for i in range(frames):
            t = i / sample_rate
            # Triple-tone alarm
            phase = int(t * 3) % 3
            if phase == 0:
                freq = 800
            elif phase == 1:
                freq = 1000
            else:
                freq = 1200
            
            drowsy_arr[i] = 0.6 * np.sin(2 * np.pi * freq * t)
        
        # Apply fade envelope
        fade_frames = frames // 20
        fade_in = np.linspace(0, 1, fade_frames)
        fade_out = np.linspace(1, 0, fade_frames)
        drowsy_arr[:fade_frames] *= fade_in
        drowsy_arr[-fade_frames:] *= fade_out
        
        drowsy_arr = (drowsy_arr * 20000).astype(np.int16)
        drowsy_stereo = np.column_stack([drowsy_arr, drowsy_arr])
        self.drowsy_sound = pygame.sndarray.make_sound(drowsy_stereo)
        
        # Face lost alert - quick double beep
        duration = 0.6
        frames = int(duration * sample_rate)
        face_arr = np.zeros(frames)
        
        # Double beep pattern
        for i in range(frames):
            t = i / sample_rate
            if t < 0.15 or (0.3 < t < 0.45):
                face_arr[i] = 0.5 * np.sin(2 * np.pi * 1500 * t)
        
        face_arr = (face_arr * 18000).astype(np.int16)
        face_stereo = np.column_stack([face_arr, face_arr])
        self.face_sound = pygame.sndarray.make_sound(face_stereo)
        
        # Configuration complete sound - pleasant chime
        chime_frames = int(0.8 * sample_rate)
        chime_arr = np.zeros(chime_frames)
        for i in range(chime_frames):
            t = i / sample_rate
            envelope = np.exp(-3 * t)
            chime_arr[i] = envelope * (0.4 * np.sin(2 * np.pi * 523 * t) + 
                                     0.3 * np.sin(2 * np.pi * 659 * t) +
                                     0.2 * np.sin(2 * np.pi * 784 * t))
        
        chime_arr = (chime_arr * 15000).astype(np.int16)
        chime_stereo = np.column_stack([chime_arr, chime_arr])
        self.config_complete_sound = pygame.sndarray.make_sound(chime_stereo)
    
    def _sound_worker(self):
        """Enhanced sound worker with priority handling"""
        while not self.shutdown:
            try:
                sound_type = self.sound_queue.get(timeout=1.0)
                if sound_type == "shutdown":
                    break
                elif sound_type == "drowsy" and hasattr(self, 'drowsy_sound'):
                    self.drowsy_sound.play()
                    pygame.time.wait(1500)  # Wait for completion
                elif sound_type == "face_lost" and hasattr(self, 'face_sound'):
                    self.face_sound.play()
                    pygame.time.wait(600)
                elif sound_type == "config_complete" and hasattr(self, 'config_complete_sound'):
                    self.config_complete_sound.play()
                    pygame.time.wait(800)
            except queue.Empty:
                continue
            except Exception as e:
                continue
    
    def play_drowsy_alert(self):
        if self.enabled and not self.shutdown:
            try:
                # Clear queue to prioritize drowsy alert
                while not self.sound_queue.empty():
                    try:
                        self.sound_queue.get_nowait()
                    except queue.Empty:
                        break
                self.sound_queue.put_nowait("drowsy")
            except queue.Full:
                pass
    
    def play_face_lost_alert(self):
        if self.enabled and not self.shutdown:
            try:
                self.sound_queue.put_nowait("face_lost")
            except queue.Full:
                pass
    
    def play_config_complete(self):
        if self.enabled and not self.shutdown:
            try:
                self.sound_queue.put_nowait("config_complete")
            except queue.Full:
                pass
    
    def shutdown_audio(self):
        self.shutdown = True
        try:
            self.sound_queue.put_nowait("shutdown")
            if hasattr(self, 'worker_thread'):
                self.worker_thread.join(timeout=2.0)
        except:
            pass

class PerformanceMonitor:
    """Advanced performance monitoring with trend analysis"""
    
    def __init__(self, window_size=60):  # 2 second window at 30fps
        self.frame_times = deque(maxlen=window_size)
        self.process_times = deque(maxlen=window_size)
        self.cnn_times = deque(maxlen=window_size)
        self.fps = 0.0
        self.avg_process_time = 0.0
        self.avg_cnn_time = 0.0
        self.max_fps = 0.0
        self.min_fps = 999.0
        
    def update_frame_time(self):
        current_time = time.time()
        self.frame_times.append(current_time)
        
        if len(self.frame_times) >= 2:
            time_diff = self.frame_times[-1] - self.frame_times[0]
            if time_diff > 0:
                self.fps = (len(self.frame_times) - 1) / time_diff
                self.max_fps = max(self.max_fps, self.fps)
                if self.fps > 1:  # Ignore very low fps readings
                    self.min_fps = min(self.min_fps, self.fps)
    
    def update_process_time(self, process_time):
        self.process_times.append(process_time)
        if self.process_times:
            self.avg_process_time = np.mean(self.process_times)
    
    def update_cnn_time(self, cnn_time):
        self.cnn_times.append(cnn_time)
        if self.cnn_times:
            self.avg_cnn_time = np.mean(self.cnn_times)
    
    def get_performance_grade(self):
        """Calculate performance grade A-F"""
        if self.fps >= 25:
            return "A"
        elif self.fps >= 20:
            return "B" 
        elif self.fps >= 15:
            return "C"
        elif self.fps >= 10:
            return "D"
        else:
            return "F"

class DrowsinessDetector:
    """Ultimate drowsiness detector with smooth operation"""
    
    def __init__(self, model, face_detector, landmark_predictor, audio_manager, manual_config=False):
        # Core components
        self.model = model
        self.face_detector = face_detector
        self.landmark_predictor = landmark_predictor
        self.audio_manager = audio_manager
        
        # Pre-compile model for GPU if available
        if GPU_AVAILABLE:
            with tf.device('/GPU:0'):
                dummy_input = tf.constant(np.zeros((1, 136, 1), dtype=np.float32))
                _ = self.model(dummy_input, training=False)
        
        # State management
        self.cnn_frame_counter = 0
        self.last_cnn_prediction = 0.0
        self.last_landmarks = None
        self.stable_prediction = 0.0
        
        # Smoothing for stable visuals
        self.prediction_history = deque(maxlen=SMOOTHING_WINDOW)
        self.confidence_history = deque(maxlen=SMOOTHING_WINDOW)
        self.ear_history = deque(maxlen=7)  # Smooth eye aspect ratio
        
        # Detection state
        self.drowsy_frame_count = 0
        self.last_alert_time = 0
        self.total_frames = 0
        self.drowsy_detections = 0
        self.eye_closed_frames = 0
        self.face_lost_frames = 0
        self.last_face_lost_alert = 0
        
        # Enhanced visual settings
        self.show_landmarks = True
        self.show_accuracy = True
        self.show_performance = True
        self.show_confidence_graph = False
        self.show_detailed_stats = False
        
        # Configuration
        self.manual_config = manual_config
        self.in_configuration = manual_config
        self.config_start_time = time.time() if manual_config else 0
        self.config_duration = MANUAL_CONFIG_DURATION
        self.calibration_complete = not manual_config
        
        # Performance monitoring
        self.perf_monitor = PerformanceMonitor()
        self.model_accuracy = 0.849
        
        # Session statistics
        self.session_start = time.time()
        self.total_alerts = 0
        self.max_consecutive_drowsy = 0
        self.current_consecutive_drowsy = 0
        
        print(f"🚀 Ultimate DrowsinessDetector v3.0 initialized")
        print(f"   📊 Model Accuracy: {self.model_accuracy:.1%}")
        print(f"   🔧 Manual Config: {'ON' if manual_config else 'OFF'}")
        print(f"   🎮 GPU Acceleration: {'RTX 4050' if GPU_AVAILABLE else 'CPU'}")
    
    def extract_landmarks_optimized(self, image):
        """Highly optimized landmark extraction"""
        try:
            # Intelligent scaling for face detection
            height, width = image.shape[:2]
            scale = FACE_DETECTION_SCALE
            small_width = int(width * scale)
            small_height = int(height * scale)
            
            small_frame = cv2.resize(image, (small_width, small_height), interpolation=cv2.INTER_LINEAR)
            gray_small = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
            
            # Face detection on scaled image
            faces = self.face_detector(gray_small, 1)
            
            if len(faces) == 0:
                return None, None
            
            # Scale back to original coordinates
            face = faces[0]
            scale_factor = 1.0 / scale
            scaled_face = dlib.rectangle(
                int(face.left() * scale_factor),
                int(face.top() * scale_factor), 
                int(face.right() * scale_factor),
                int(face.bottom() * scale_factor)
            )
            
            # Extract landmarks on full resolution
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            landmarks = self.landmark_predictor(gray, scaled_face)
            
            coords = np.zeros(136, dtype=np.float32)
            landmark_points = np.zeros((68, 2), dtype=np.float32)
            
            for i in range(68):
                x, y = landmarks.part(i).x, landmarks.part(i).y
                coords[i*2] = x
                coords[i*2+1] = y
                landmark_points[i] = [x, y]
            
            return coords, landmark_points
        
        except Exception:
            return None, None
    
    def normalize_landmarks_fast(self, landmarks):
        """Vectorized landmark normalization"""
        x = landmarks[::2]
        y = landmarks[1::2]
        
        x_min, x_max = x.min(), x.max()
        y_min, y_max = y.min(), y.max()
        
        x_range = x_max - x_min + 1e-6
        y_range = y_max - y_min + 1e-6
        
        landmarks[::2] = (x - x_min) / x_range
        landmarks[1::2] = (y - y_min) / y_range
        
        return landmarks
    
    def predict_drowsiness_gpu_optimized(self, landmarks):
        """GPU-optimized prediction with caching"""
        try:
            start_time = time.time()
            
            # Fast normalization
            normalized = self.normalize_landmarks_fast(landmarks.copy())
            
            # Prepare input tensor
            features = tf.constant(normalized.reshape(1, 136, 1), dtype=tf.float32)
            
            # GPU prediction with explicit device placement
            if GPU_AVAILABLE:
                with tf.device('/GPU:0'):
                    prediction = self.model(features, training=False)
            else:
                prediction = self.model(features, training=False)
            
            result = float(prediction.numpy()[0][1])
            
            cnn_time = (time.time() - start_time) * 1000
            self.perf_monitor.update_cnn_time(cnn_time)
            
            return result
            
        except Exception:
            return 0.0
    
    def analyze_eye_closure_enhanced(self, landmark_points):
        """Enhanced eye analysis with smoothing"""
        if landmark_points is None or len(landmark_points) < 68:
            return False, 0.0
        
        try:
            left_eye = landmark_points[LEFT_EYE]
            right_eye = landmark_points[RIGHT_EYE]
            
            left_ear = calculate_eye_aspect_ratio(left_eye)
            right_ear = calculate_eye_aspect_ratio(right_eye)
            
            avg_ear = (left_ear + right_ear) * 0.5
            
            # Smooth EAR readings
            self.ear_history.append(avg_ear)
            smoothed_ear = np.mean(self.ear_history) if self.ear_history else avg_ear
            
            return smoothed_ear < EYE_AR_THRESHOLD, smoothed_ear
            
        except Exception:
            return False, 0.0
    
    def draw_professional_overlay(self, frame, prob, status, color, confidence):
        """Professional overlay with smooth animations"""
        overlay_height = 180
        overlay_y_start = frame.shape[0] - overlay_height
        
        # Create gradient overlay
        overlay = np.zeros((overlay_height, frame.shape[1], 3), dtype=np.uint8)
        for i in range(overlay_height):
            alpha = 1.0 - (i / overlay_height) * 0.3
            overlay[i, :] = (0, 0, 0)
        
        # Apply overlay
        frame[overlay_y_start:, :] = cv2.addWeighted(
            frame[overlay_y_start:, :], 0.3, overlay, 0.7, 0
        )
        
        # Add border line
        cv2.line(frame, (0, overlay_y_start), (frame.shape[1], overlay_y_start), 
                COLORS['blue'], 3)
        
        y_pos = overlay_y_start + 30
        
        # Main status with confidence
        status_text = f"{status}"
        cv2.putText(frame, status_text, (15, y_pos), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        # Confidence bar
        conf_width = int(confidence * 200)
        cv2.rectangle(frame, (15, y_pos + 10), (215, y_pos + 25), COLORS['gray'], 2)
        cv2.rectangle(frame, (15, y_pos + 10), (15 + conf_width, y_pos + 25), color, -1)
        cv2.putText(frame, f"{confidence:.0%}", (225, y_pos + 22), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS['white'], 1)
        
        y_pos += 45
        
        # Performance metrics
        if self.show_performance:
            grade = self.perf_monitor.get_performance_grade()
            grade_color = COLORS['green'] if grade in ['A', 'B'] else COLORS['yellow'] if grade == 'C' else COLORS['red']
            
            perf_text = f"FPS: {self.perf_monitor.fps:.1f} | Grade: {grade} | Process: {self.perf_monitor.avg_process_time:.1f}ms"
            cv2.putText(frame, perf_text, (15, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, grade_color, 1)
            y_pos += 25
            
            if GPU_AVAILABLE and self.perf_monitor.avg_cnn_time > 0:
                gpu_text = f"GPU Inference: {self.perf_monitor.avg_cnn_time:.1f}ms"
                cv2.putText(frame, gpu_text, (15, y_pos), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS['blue'], 1)
                y_pos += 25
        
        # Model accuracy
        if self.show_accuracy:
            accuracy_text = f"Model Accuracy: {self.model_accuracy:.1%} | Session Alerts: {self.total_alerts}"
            cv2.putText(frame, accuracy_text, (15, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS['yellow'], 1)
            y_pos += 25
        
        # Session statistics
        if self.show_detailed_stats:
            session_time = time.time() - self.session_start
            stats_text = f"Session: {session_time/60:.1f}min | Max Streak: {self.max_consecutive_drowsy}"
            cv2.putText(frame, stats_text, (15, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLORS['gray'], 1)
            y_pos += 20
        
        # Status indicator (top right)
        status_x = frame.shape[1] - 150
        if self.in_configuration:
            cv2.putText(frame, "🔧 CALIBRATING", (status_x, overlay_y_start + 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLORS['yellow'], 2)
        else:
            cv2.putText(frame, "✅ MONITORING", (status_x, overlay_y_start + 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLORS['green'], 2)
        
        # Enhanced controls
        controls = [
            "D=Landmarks  A=Accuracy  P=Performance",
            "C=Configure  R=Reset Stats  Q=Quit  S=Screenshot"
        ]
        for i, control in enumerate(controls):
            cv2.putText(frame, control, (status_x - 100, overlay_y_start + 60 + i*20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.35, COLORS['white'], 1)
    
    def process_frame_ultimate(self, frame):
        """Ultimate frame processing with no flickering"""
        process_start = time.time()
        self.total_frames += 1
        
        # Configuration phase handling
        if self.in_configuration:
            elapsed = time.time() - self.config_start_time
            if elapsed >= self.config_duration:
                self.in_configuration = False
                self.calibration_complete = True
                self.audio_manager.play_config_complete()
                print("✅ Configuration complete! Enhanced detection active.")
            else:
                remaining = self.config_duration - elapsed
                self.draw_config_overlay_enhanced(frame, remaining)
                return frame, 0.0, "Calibrating...", COLORS['yellow'], 0.0
        
        # Extract landmarks
        landmarks, landmark_points = self.extract_landmarks_optimized(frame)
        
        if landmarks is None:
            self.face_lost_frames += 1
            if self.face_lost_frames > FACE_LOST_THRESHOLD and not self.in_configuration:
                self.trigger_face_lost_alert()
            return frame, 0.0, "Face not detected", COLORS['yellow'], 0.0
        
        self.face_lost_frames = 0
        self.last_landmarks = landmark_points
        
        # Enhanced eye analysis
        eyes_closed, eye_ratio = self.analyze_eye_closure_enhanced(landmark_points)
        
        if eyes_closed:
            self.eye_closed_frames += 1
        else:
            self.eye_closed_frames = 0
        
        # Smart CNN prediction
        self.cnn_frame_counter += 1
        if self.cnn_frame_counter >= CNN_PREDICTION_INTERVAL:
            self.last_cnn_prediction = self.predict_drowsiness_gpu_optimized(landmarks)
            self.cnn_frame_counter = 0
        
        # Smooth predictions
        drowsy_prob = self.last_cnn_prediction
        self.prediction_history.append(drowsy_prob)
        smoothed_prob = np.mean(self.prediction_history)
        
        # Enhanced detection logic
        cnn_drowsy = smoothed_prob > DROWSINESS_THRESHOLD
        eye_drowsy = self.eye_closed_frames >= EYE_AR_CONSEC_FRAMES
        is_drowsy = cnn_drowsy or eye_drowsy
        
        # Calculate enhanced confidence
        if is_drowsy:
            if eye_drowsy:
                confidence = min(1.0, (EYE_AR_THRESHOLD - eye_ratio + 0.1) / 0.15)
                status = f"EYES CLOSED! EAR: {eye_ratio:.3f}"
                color = COLORS['red']
            else:
                confidence = smoothed_prob
                status = f"DROWSINESS DETECTED! CNN: {smoothed_prob:.3f}"
                color = COLORS['red']
            
            self.drowsy_frame_count += 1
            self.current_consecutive_drowsy += 1
            self.max_consecutive_drowsy = max(self.max_consecutive_drowsy, 
                                            self.current_consecutive_drowsy)
            
            if self.drowsy_frame_count >= CONSECUTIVE_FRAMES:
                self.trigger_drowsiness_alert()
                
        else:
            confidence = 1.0 - smoothed_prob
            status = f"ALERT - CNN: {smoothed_prob:.3f} | EAR: {eye_ratio:.3f}"
            color = COLORS['green']
            self.drowsy_frame_count = 0
            self.current_consecutive_drowsy = 0
        
        # Update confidence history for smoothing
        self.confidence_history.append(confidence)
        smooth_confidence = np.mean(self.confidence_history)
        
        # Draw enhanced landmarks
        if self.show_landmarks and landmark_points is not None:
            self.draw_landmarks_enhanced(frame, landmark_points, is_drowsy)
        
        # Update performance metrics
        process_time = (time.time() - process_start) * 1000
        self.perf_monitor.update_process_time(process_time)
        
        return frame, smoothed_prob, status, color, smooth_confidence
    
    def draw_config_overlay_enhanced(self, frame, remaining):
        """Enhanced configuration overlay"""
        # Semi-transparent background
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (frame.shape[1], 150), (0, 0, 0), -1)
        frame = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)
        
        # Progress bar
        progress = 1.0 - (remaining / self.config_duration)
        bar_width = int(400 * progress)
        cv2.rectangle(frame, (frame.shape[1]//2 - 200, 80), 
                     (frame.shape[1]//2 + 200, 100), COLORS['gray'], 2)
        cv2.rectangle(frame, (frame.shape[1]//2 - 200, 80), 
                     (frame.shape[1]//2 - 200 + bar_width, 100), COLORS['blue'], -1)
        
        # Text
        text = f"CALIBRATION: {remaining:.1f}s remaining ({progress:.0%})"
        cv2.putText(frame, text, (frame.shape[1]//2 - 200, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, COLORS['yellow'], 2)
        
        instruction = "Position your face in the center guide below"
        cv2.putText(frame, instruction, (frame.shape[1]//2 - 200, 120), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLORS['white'], 1)
        
        # Enhanced center guide
        center_x, center_y = frame.shape[1]//2, frame.shape[0]//2 - 50
        cv2.circle(frame, (center_x, center_y), 100, COLORS['green'], 3)
        cv2.circle(frame, (center_x, center_y), 80, COLORS['blue'], 2)
        cv2.line(frame, (center_x-20, center_y), (center_x+20, center_y), COLORS['green'], 3)
        cv2.line(frame, (center_x, center_y-20), (center_x, center_y+20), COLORS['green'], 3)
    
    def draw_landmarks_enhanced(self, frame, landmark_points, is_drowsy):
        """Enhanced landmark drawing with drowsiness indication"""
        # Eye landmarks with special highlighting
        for i, (x, y) in enumerate(landmark_points):
            if i in LEFT_EYE + RIGHT_EYE:
                color = COLORS['red'] if is_drowsy else COLORS['green']
                size = 4 if is_drowsy else 3
                cv2.circle(frame, (int(x), int(y)), size, color, -1)
                cv2.circle(frame, (int(x), int(y)), size + 1, COLORS['white'], 1)
            else:
                cv2.circle(frame, (int(x), int(y)), 2, COLORS['blue'], -1)
        
        # Enhanced eye contours
        try:
            left_eye_hull = cv2.convexHull(landmark_points[LEFT_EYE].astype(int))
            right_eye_hull = cv2.convexHull(landmark_points[RIGHT_EYE].astype(int))
            
            eye_color = COLORS['red'] if is_drowsy else COLORS['green']
            thickness = 3 if is_drowsy else 2
            
            cv2.drawContours(frame, [left_eye_hull], -1, eye_color, thickness)
            cv2.drawContours(frame, [right_eye_hull], -1, eye_color, thickness)
            
            if is_drowsy:
                # Add pulsing effect for drowsy state
                cv2.drawContours(frame, [left_eye_hull], -1, COLORS['yellow'], 1)
                cv2.drawContours(frame, [right_eye_hull], -1, COLORS['yellow'], 1)
                
        except Exception:
            pass
    
    def trigger_drowsiness_alert(self):
        current_time = time.time()
        if current_time - self.last_alert_time > ALERT_COOLDOWN:
            print("🚨 ULTIMATE DROWSINESS ALERT! 🚨")
            self.audio_manager.play_drowsy_alert()
            self.last_alert_time = current_time
            self.total_alerts += 1
    
    def trigger_face_lost_alert(self):
        current_time = time.time()
        if current_time - self.last_face_lost_alert > 1:  # Longer cooldown
            print("⚠️ FACE NOT DETECTED - Please look at camera!")
            self.audio_manager.play_face_lost_alert()
            self.last_face_lost_alert = current_time
    
    def toggle_landmarks(self):
        self.show_landmarks = not self.show_landmarks
        print(f"🎯 Enhanced Landmarks: {'ON' if self.show_landmarks else 'OFF'}")
    
    def toggle_accuracy_display(self):
        self.show_accuracy = not self.show_accuracy
        print(f"📊 Accuracy Display: {'ON' if self.show_accuracy else 'OFF'}")
    
    def toggle_performance_display(self):
        self.show_performance = not self.show_performance
        print(f"⚡ Performance Display: {'ON' if self.show_performance else 'OFF'}")
    
    def toggle_detailed_stats(self):
        self.show_detailed_stats = not self.show_detailed_stats
        print(f"📈 Detailed Stats: {'ON' if self.show_detailed_stats else 'OFF'}")
    
    def start_manual_config(self):
        self.in_configuration = True
        self.config_start_time = time.time()
        print("🔧 Enhanced manual configuration started")
    
    def reset_session_stats(self):
        self.total_alerts = 0
        self.max_consecutive_drowsy = 0
        self.current_consecutive_drowsy = 0
        self.session_start = time.time()
        self.total_frames = 0
        self.drowsy_detections = 0
        print("📊 Session statistics reset")
    
    def get_enhanced_statistics(self):
        if self.total_frames == 0:
            return "No frames processed"
        
        drowsy_percentage = (self.drowsy_detections / self.total_frames) * 100
        session_time = time.time() - self.session_start
        
        return (f"Frames: {self.total_frames} | Drowsy: {drowsy_percentage:.1f}% | "
                f"Alerts: {self.total_alerts} | Session: {session_time/60:.1f}min")

def cleanup():
    """Enhanced cleanup with status reporting"""
    global cap, audio_manager
    
    print("🧹 Cleaning up resources...")
    
    if audio_manager:
        audio_manager.shutdown_audio()
        print("   🔊 Audio system shutdown")
    
    if cap and cap.isOpened():
        cap.release()
        print("   📷 Camera released")
    
    cv2.destroyAllWindows()
    print("   🖼️ Windows closed")
    print("✅ Cleanup completed successfully")

def parse_arguments():
    parser = argparse.ArgumentParser(description='Ultimate Drowsiness Detection System v3.0')
    parser.add_argument('--manual-config', action='store_true', 
                       help='Enable manual configuration phase')
    parser.add_argument('--camera-index', type=int, default=0,
                       help='Camera index (default: 0)')
    parser.add_argument('--no-audio', action='store_true',
                       help='Disable audio alerts')
    parser.add_argument('--force-cpu', action='store_true',
                       help='Force CPU usage (disable GPU)')
    return parser.parse_args()

def main():
    """Ultimate main application"""
    global cap, audio_manager
    
    args = parse_arguments()
    
    if args.force_cpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
        print("🔧 GPU acceleration disabled (forced CPU mode)")
    
    print("🚀 ULTIMATE DROWSINESS DETECTION SYSTEM v3.0")
    print("="*70)
    print("🎯 Enhanced Features:")
    print("   ⚡ RTX 4050 GPU Acceleration")
    print("   📊 Real-time Performance Analytics") 
    print("   🔧 Professional Configuration System")
    print("   📈 Live Accuracy & Confidence Display")
    print("   🎮 Advanced Interactive Controls")
    print("   🔊 Professional Audio Alert System")
    print("   📱 Session Statistics & Trend Analysis")
    print("="*70)
    
    atexit.register(cleanup)
    
    try:
        # Load and optimize model
        print("🔄 Loading and optimizing model...")
        model = load_model(MODEL_PATH)
        
        if GPU_AVAILABLE:
            # Warm up GPU
            with tf.device('/GPU:0'):
                dummy_input = tf.constant(np.zeros((1, 136, 1), dtype=tf.float32))
                for _ in range(3):  # Multiple warmup runs
                    _ = model(dummy_input, training=False)
            print("✅ Model loaded and GPU optimized")
        else:
            # CPU warmup
            dummy_input = np.zeros((1, 136, 1), dtype=np.float32)
            model.predict(dummy_input, verbose=0)
            print("✅ Model loaded (CPU mode)")
        
        # Load face detection
        face_detector = dlib.get_frontal_face_detector()
        landmark_predictor = dlib.shape_predictor(PREDICTOR_PATH)
        print("✅ Face detection components loaded")
        
        # Initialize enhanced audio
        if not args.no_audio:
            audio_manager = AudioManager()
        else:
            audio_manager = None
            print("🔇 Audio system disabled")
        
        # Initialize optimized camera
        cap = cv2.VideoCapture(args.camera_index, cv2.CAP_DSHOW)  # Windows optimization
        if not cap.isOpened():
            raise Exception(f"Cannot open camera {args.camera_index}")
        
        # Optimized camera settings
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        # Let camera use auto settings for stable operation
        
        print(f"✅ Camera {args.camera_index} initialized with optimizations")
        
        # Initialize ultimate detector
        detector = DrowsinessDetector(model, face_detector, landmark_predictor, 
                                    audio_manager, args.manual_config)
        
        print("\n🎮 ULTIMATE CONTROLS:")
        print("   D = Toggle Enhanced Landmarks    A = Toggle Accuracy Display")
        print("   P = Toggle Performance Monitor   R = Reset Session Stats")
        print("   C = Manual Configuration         T = Toggle Detailed Stats")
        print("   Q = Quit Application            S = Save Screenshot")
        print("\n🚀 Starting ultimate detection system...")
        
        # Main processing loop
        while True:
            ret, frame = cap.read()
            if not ret:
                print("❌ Failed to read from camera")
                break
            
            # Mirror for natural interaction
            frame = cv2.flip(frame, 1)
            detector.perf_monitor.update_frame_time()
            
            # Ultimate frame processing
            try:
                processed_frame, prob, status, color, confidence = detector.process_frame_ultimate(frame)
                detector.draw_professional_overlay(processed_frame, prob, status, color, confidence)
                
            except Exception as e:
                print(f"⚠️ Processing error: {e}")
                processed_frame = frame
            
            # Display with enhanced title
            cv2.imshow('🚀 Ultimate Drowsiness Detection System v3.0 - RTX 4050 Enhanced', processed_frame)
            
            # Enhanced controls
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == ord('Q'):
                print("👋 Shutting down ultimate system...")
                break
            elif key == ord('d') or key == ord('D'):
                detector.toggle_landmarks()
            elif key == ord('a') or key == ord('A'):
                detector.toggle_accuracy_display()
            elif key == ord('p') or key == ord('P'):
                detector.toggle_performance_display()
            elif key == ord('c') or key == ord('C'):
                detector.start_manual_config()
            elif key == ord('r') or key == ord('R'):
                detector.reset_session_stats()
            elif key == ord('t') or key == ord('T'):
                detector.toggle_detailed_stats()
            elif key == ord('s') or key == ord('S'):
                filename = f"ultimate_detection_{int(time.time())}.jpg"
                cv2.imwrite(filename, processed_frame)
                print(f"📸 Ultimate screenshot saved: {filename}")
    
    except KeyboardInterrupt:
        print("\n⚠️ System interrupted by user")
    except Exception as e:
        print(f"❌ Critical error: {e}")
    finally:
        print("\n📊 Ultimate Session Report:")
        if 'detector' in locals():
            print(detector.get_enhanced_statistics())
            print(f"Performance Grade: {detector.perf_monitor.get_performance_grade()}")
            print(f"FPS Range: {detector.perf_monitor.min_fps:.1f} - {detector.perf_monitor.max_fps:.1f}")
        print("🎉 Ultimate detection session completed!")

if __name__ == "__main__":
    main()
