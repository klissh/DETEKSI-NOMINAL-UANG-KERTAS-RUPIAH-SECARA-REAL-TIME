import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk, Label, Frame, Scale, HORIZONTAL, Text, Scrollbar
from PIL import Image, ImageTk
from ultralytics import YOLO
import threading
import time
from playsound import playsound
import os


class MoneyDetectorApp:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Indonesian Money Detector - Enhanced")
        self.root.geometry("1600x900")
        self.root.configure(bg="#f5f5f5")
        
        # Style configuration
        self.style = ttk.Style()
        self.style.theme_use('clam')
        self.style.configure('TFrame', background='#f5f5f5')
        self.style.configure('TLabel', background='#f5f5f5', font=('Arial', 10))
        
        # Model and camera setup
        self.cap = cv2.VideoCapture(0)
        # Set camera properties for better detection
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        
        # Track last detected denomination to avoid repeated audio
        self.last_detected_nominal = None
        self.last_audio_time = 0
        self.audio_cooldown = 3  # seconds between audio playbacks
        
        try:
            self.model = YOLO("E:/documents/Kuliah/Semester 4/Pengolahan Citra/Project UAS/train/weights/best.pt")
            print("Model loaded successfully")
            print(f"Model classes: {self.model.names}")
        except Exception as e:
            print(f"Error loading model: {e}")
            self.model = None
        
        self.detection_active = True
        self.debug_mode = True
        
        # Enhanced parameters for better detection
        self.params = {
            'conf_threshold': 0.25,  # Lower threshold for initial detection
            'iou_threshold': 0.45,
            'imgsz': 640,
            'max_det': 10,
            'agnostic_nms': False,
            'enhance_before_detection': True,
            'multi_scale_detection': True,
            'brightness_adjustment': 0,
            'contrast_adjustment': 1.0,
            'gamma_correction': 1.0
        }
        
        self.setup_ui()
        self.run_detection_loop()
    
    def setup_ui(self):
        # Main container
        main_container = ttk.Frame(self.root)
        main_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Left panel (camera and debug)
        left_panel = ttk.Frame(main_container)
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Camera feed frame
        camera_frame = ttk.Frame(left_panel)
        camera_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        ttk.Label(camera_frame, text="Live Detection Feed", font=('Arial', 14, 'bold')).pack(pady=5)
        self.label_webcam = ttk.Label(camera_frame)
        self.label_webcam.pack(pady=5)
        
        # Enhanced camera feed (preprocessed)
        enhanced_frame = ttk.Frame(left_panel)
        enhanced_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(enhanced_frame, text="Enhanced Feed", font=('Arial', 12, 'bold')).pack()
        self.label_enhanced = ttk.Label(enhanced_frame)
        self.label_enhanced.pack()
        
        # Debug information
        debug_frame = ttk.LabelFrame(left_panel, text="Debug Information")
        debug_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # Create text widget with scrollbar for debug info
        text_frame = ttk.Frame(debug_frame)
        text_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.debug_text = Text(text_frame, height=8, wrap=tk.WORD, font=('Courier', 9))
        scrollbar = Scrollbar(text_frame, orient=tk.VERTICAL, command=self.debug_text.yview)
        self.debug_text.configure(yscrollcommand=scrollbar.set)
        
        self.debug_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Right panel (controls and results)
        right_panel = ttk.Frame(main_container)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, padx=(10, 0))
        
        # Results frame
        results_frame = ttk.LabelFrame(right_panel, text="Detection Results")
        results_frame.pack(fill=tk.X, pady=5)
        
        self.status_indicator = ttk.Label(results_frame, text="●", font=('Arial', 24), foreground='gray')
        self.status_indicator.pack(pady=5)
        
        self.result_label = ttk.Label(results_frame, text="Waiting for detection...", 
                                     font=('Arial', 14, 'bold'))
        self.result_label.pack(pady=5)
        
        self.confidence_label = ttk.Label(results_frame, text="Confidence: --", 
                                         font=('Arial', 12))
        self.confidence_label.pack()
        
        self.detection_count_label = ttk.Label(results_frame, text="Objects detected: 0", 
                                              font=('Arial', 10))
        self.detection_count_label.pack()
        
        # Audio status
        self.audio_status_label = ttk.Label(results_frame, text="Audio: Ready", 
                                           font=('Arial', 10), foreground='blue')
        self.audio_status_label.pack(pady=5)
        
        # Detection controls
        controls_frame = ttk.LabelFrame(right_panel, text="Detection Controls")
        controls_frame.pack(fill=tk.X, pady=5)
        
        # Confidence threshold
        ttk.Label(controls_frame, text="Confidence Threshold:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=2)
        conf_scale = Scale(controls_frame, from_=0.05, to=0.95, resolution=0.05, orient=HORIZONTAL,
                          command=lambda v: self.update_param('conf_threshold', float(v)))
        conf_scale.set(self.params['conf_threshold'])
        conf_scale.grid(row=0, column=1, sticky=tk.EW, padx=5, pady=2)
        
        # Image size
        ttk.Label(controls_frame, text="Detection Image Size:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=2)
        size_scale = Scale(controls_frame, from_=320, to=1280, resolution=32, orient=HORIZONTAL,
                          command=lambda v: self.update_param('imgsz', int(v)))
        size_scale.set(self.params['imgsz'])
        size_scale.grid(row=1, column=1, sticky=tk.EW, padx=5, pady=2)
        
        # Brightness adjustment
        ttk.Label(controls_frame, text="Brightness:").grid(row=2, column=0, sticky=tk.W, padx=5, pady=2)
        brightness_scale = Scale(controls_frame, from_=-100, to=100, resolution=5, orient=HORIZONTAL,
                                command=lambda v: self.update_param('brightness_adjustment', int(v)))
        brightness_scale.set(self.params['brightness_adjustment'])
        brightness_scale.grid(row=2, column=1, sticky=tk.EW, padx=5, pady=2)
        
        # Contrast adjustment
        ttk.Label(controls_frame, text="Contrast:").grid(row=3, column=0, sticky=tk.W, padx=5, pady=2)
        contrast_scale = Scale(controls_frame, from_=0.5, to=3.0, resolution=0.1, orient=HORIZONTAL,
                              command=lambda v: self.update_param('contrast_adjustment', float(v)))
        contrast_scale.set(self.params['contrast_adjustment'])
        contrast_scale.grid(row=3, column=1, sticky=tk.EW, padx=5, pady=2)
        
        # Gamma correction
        ttk.Label(controls_frame, text="Gamma:").grid(row=4, column=0, sticky=tk.W, padx=5, pady=2)
        gamma_scale = Scale(controls_frame, from_=0.5, to=2.5, resolution=0.1, orient=HORIZONTAL,
                           command=lambda v: self.update_param('gamma_correction', float(v)))
        gamma_scale.set(self.params['gamma_correction'])
        gamma_scale.grid(row=4, column=1, sticky=tk.EW, padx=5, pady=2)
        
        controls_frame.columnconfigure(1, weight=1)
        
        # Action buttons
        btn_frame = ttk.Frame(right_panel)
        btn_frame.pack(fill=tk.X, pady=5)
        
        self.toggle_btn = ttk.Button(btn_frame, text="Pause Detection", command=self.toggle_detection)
        self.toggle_btn.pack(side=tk.LEFT, padx=2)
        
        self.debug_btn = ttk.Button(btn_frame, text="Toggle Debug", command=self.toggle_debug)
        self.debug_btn.pack(side=tk.LEFT, padx=2)
        
        reset_btn = ttk.Button(btn_frame, text="Reset", command=self.reset_params)
        reset_btn.pack(side=tk.LEFT, padx=2)
        
        # Audio toggle button
        self.audio_enabled = True
        self.audio_btn = ttk.Button(btn_frame, text="Mute Audio", command=self.toggle_audio)
        self.audio_btn.pack(side=tk.LEFT, padx=2)
        
        # Processing preview
        preview_frame = ttk.LabelFrame(right_panel, text="Processing Steps")
        preview_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        self.label_original = self.add_small_preview(preview_frame, "Original", 0, 0)
        self.label_processed = self.add_small_preview(preview_frame, "Processed", 0, 1)
        self.label_contrast = self.add_small_preview(preview_frame, "Contrast", 1, 0)
        self.label_final = self.add_small_preview(preview_frame, "Final", 1, 1)
    
    def add_small_preview(self, parent, text, row, col):
        frame = ttk.Frame(parent)
        frame.grid(row=row*2, column=col, padx=5, pady=2, sticky=tk.W)
        
        ttk.Label(frame, text=text, font=('Arial', 9)).pack()
        label = ttk.Label(frame, borderwidth=1, relief="solid")
        label.pack()
        return label
    
    def update_param(self, param_name, value):
        self.params[param_name] = value
        self.log_debug(f"Updated {param_name}: {value}")
    
    def toggle_detection(self):
        self.detection_active = not self.detection_active
        self.toggle_btn.config(text="Resume Detection" if not self.detection_active else "Pause Detection")
        self.log_debug(f"Detection {'paused' if not self.detection_active else 'resumed'}")
    
    def toggle_debug(self):
        self.debug_mode = not self.debug_mode
        self.debug_btn.config(text="Debug: ON" if self.debug_mode else "Debug: OFF")
    
    def toggle_audio(self):
        self.audio_enabled = not self.audio_enabled
        self.audio_btn.config(text="Enable Audio" if not self.audio_enabled else "Mute Audio")
        self.audio_status_label.config(text=f"Audio: {'Enabled' if self.audio_enabled else 'Muted'}")
        self.log_debug(f"Audio {'disabled' if not self.audio_enabled else 'enabled'}")
    
    def reset_params(self):
        self.params.update({
            'conf_threshold': 0.25,
            'brightness_adjustment': 0,
            'contrast_adjustment': 1.0,
            'gamma_correction': 1.0
        })
        self.log_debug("Parameters reset to defaults")
    
    def log_debug(self, message):
        if self.debug_mode:
            timestamp = time.strftime("%H:%M:%S")
            self.debug_text.insert(tk.END, f"[{timestamp}] {message}\n")
            self.debug_text.see(tk.END)
            
    def play_audio_for_nominal(self, nominal_value):
        if not self.audio_enabled:
            return
            
        # Check if we should play audio (avoid repeating too frequently)
        current_time = time.time()
        if (nominal_value == self.last_detected_nominal and 
            current_time - self.last_audio_time < self.audio_cooldown):
            return
            
        self.last_detected_nominal = nominal_value
        self.last_audio_time = current_time
        
        # Update audio status
        self.audio_status_label.config(text=f"Audio: Playing {nominal_value}", foreground='green')
        self.root.update_idletasks()  # Force UI update
        
        audio_file = os.path.join("Audio", f"{nominal_value}.mp3")
        self.log_debug(f"Checking for audio: {audio_file}")
        
        if os.path.exists(audio_file):
            try:
                self.log_debug(f"Audio found. Playing: {audio_file}")
                threading.Thread(target=self._play_sound, args=(audio_file,), daemon=True).start()
            except Exception as e:
                self.log_debug(f"Error playing audio: {e}")
                self.audio_status_label.config(text="Audio: Error", foreground='red')
        else:
            self.log_debug(f"Audio not found: {audio_file}")
            self.audio_status_label.config(text="Audio: File not found", foreground='orange')
    
    def _play_sound(self, audio_file):
        try:
            playsound(audio_file)
            # Reset audio status after playing
            self.root.after(1000, lambda: self.audio_status_label.config(text="Audio: Ready", foreground='blue'))
        except Exception as e:
            self.log_debug(f"Error in audio playback thread: {e}")
            self.root.after(0, lambda: self.audio_status_label.config(text="Audio: Error", foreground='red'))
    
    def enhance_image(self, image):
        """Apply image enhancements to improve detection"""
        enhanced = image.copy()
        
        # Brightness adjustment
        if self.params['brightness_adjustment'] != 0:
            enhanced = cv2.convertScaleAbs(enhanced, alpha=1, beta=self.params['brightness_adjustment'])
        
        # Contrast adjustment
        if self.params['contrast_adjustment'] != 1.0:
            enhanced = cv2.convertScaleAbs(enhanced, alpha=self.params['contrast_adjustment'], beta=0)
        
        # Gamma correction
        if self.params['gamma_correction'] != 1.0:
            gamma = self.params['gamma_correction']
            inv_gamma = 1.0 / gamma
            table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
            enhanced = cv2.LUT(enhanced, table)
        
        # Additional enhancement techniques
        # Histogram equalization on LAB color space
        lab = cv2.cvtColor(enhanced, cv2.COLOR_BGR2LAB)
        lab[:,:,0] = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)).apply(lab[:,:,0])
        enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        
        return enhanced
    
    def run_detection_loop(self):
        def loop():
            frame_count = 0
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    self.log_debug("Failed to read frame from camera")
                    time.sleep(0.1)
                    continue
                
                frame_count += 1
                
                if not self.detection_active:
                    time.sleep(0.1)
                    continue
                
                if self.model is None:
                    self.log_debug("Model not loaded")
                    time.sleep(1)
                    continue
                
                try:
                    # Create a copy for drawing
                    frame_copy = frame.copy()
                    
                    # Enhance image for better detection
                    enhanced_frame = self.enhance_image(frame)
                    
                    # Multiple detection attempts with different parameters
                    all_detections = []
                    
                    # Detection attempt 1: Original enhanced frame
                    results1 = self.model.predict(
                        enhanced_frame, 
                        imgsz=self.params['imgsz'], 
                        conf=self.params['conf_threshold'],
                        iou=self.params['iou_threshold'],
                        max_det=self.params['max_det'],
                        agnostic_nms=self.params['agnostic_nms'],
                        verbose=False
                    )
                    
                    if len(results1[0].boxes) > 0:
                        all_detections.extend(results1[0].boxes)
                    
                    # Detection attempt 2: Different image size if multi-scale is enabled
                    if self.params['multi_scale_detection'] and self.params['imgsz'] != 416:
                        results2 = self.model.predict(
                            enhanced_frame, 
                            imgsz=416, 
                            conf=self.params['conf_threshold'] * 0.8,  # Slightly lower threshold
                            verbose=False
                        )
                        if len(results2[0].boxes) > 0:
                            all_detections.extend(results2[0].boxes)
                    
                    # Detection attempt 3: Grayscale conversion
                    gray_frame = cv2.cvtColor(enhanced_frame, cv2.COLOR_BGR2GRAY)
                    gray_bgr = cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2BGR)
                    results3 = self.model.predict(
                        gray_bgr, 
                        imgsz=self.params['imgsz'], 
                        conf=self.params['conf_threshold'] * 0.9,
                        verbose=False
                    )
                    if len(results3[0].boxes) > 0:
                        all_detections.extend(results3[0].boxes)
                    
                    # Process all detections
                    if all_detections:
                        self.log_debug(f"Frame {frame_count}: Found {len(all_detections)} total detections")
                        
                        # Convert to numpy arrays for processing
                        boxes_list = []
                        confs_list = []
                        classes_list = []
                        
                        for detection in all_detections:
                            if hasattr(detection, 'xyxy') and hasattr(detection, 'conf') and hasattr(detection, 'cls'):
                                boxes_list.append(detection.xyxy.cpu().numpy())
                                confs_list.append(detection.conf.cpu().numpy())
                                classes_list.append(detection.cls.cpu().numpy())
                        
                        if boxes_list:
                            # Find the best detection
                            best_conf = 0
                            best_detection = None
                            best_class = None
                            
                            for i, (boxes, confs, classes) in enumerate(zip(boxes_list, confs_list, classes_list)):
                                for j, (box, conf, cls) in enumerate(zip(boxes, confs, classes)):
                                    if conf > best_conf:
                                        best_conf = conf
                                        best_detection = box
                                        best_class = int(cls)
                            
                            if best_detection is not None:
                                x1, y1, x2, y2 = best_detection.astype(int)
                                
                                # Draw bounding box
                                cv2.rectangle(frame_copy, (x1, y1), (x2, y2), (0, 255, 0), 3)
                                
                                # Get class name
                                class_name = self.model.names.get(best_class, f"Class_{best_class}")
                                
                                # Draw label with confidence
                                label_text = f"{class_name}: {best_conf:.2f}"
                                cv2.putText(frame_copy, label_text, (x1, y1-10), 
                                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                                
                                # Format denomination and play audio
                                try:
                                    nominal = int(class_name.replace('.', '').replace(',', ''))
                                    formatted_label = f'Rp{nominal:,}'.replace(',', '.').replace('.', ',', 1)
                                    
                                    # Play audio for the detected denomination
                                    self.play_audio_for_nominal(nominal)
                                except Exception as e:
                                    formatted_label = class_name
                                    self.log_debug(f"Failed to parse nominal from class_name: {class_name} | Error: {e}")
                                
                                # Update UI
                                self.result_label.config(text=f"Detected: {formatted_label}")
                                self.confidence_label.config(text=f"Confidence: {best_conf:.3f}")
                                self.detection_count_label.config(text=f"Objects detected: {len(all_detections)}")
                                self.status_indicator.config(foreground='green')
                                
                                self.log_debug(f"Detection: {formatted_label} (conf: {best_conf:.3f})")
                                
                                # Process the detected region for preview
                                if x2 > x1 and y2 > y1:
                                    crop = frame[y1:y2, x1:x2]
                                    if crop.size > 0:
                                        # Apply processing steps for preview
                                        crop_gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                                        
                                        # Contrast enhancement
                                        min_val = np.percentile(crop_gray, 2)
                                        max_val = np.percentile(crop_gray, 98)
                                        if max_val > min_val:
                                            contrast = np.clip((crop_gray - min_val) * 255.0 / (max_val - min_val), 0, 255).astype(np.uint8)
                                        else:
                                            contrast = crop_gray
                                        
                                        # Final processing
                                        sharpen_kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
                                        final = cv2.filter2D(contrast, -1, sharpen_kernel)
                                        
                                        # Update preview images
                                        self.update_small_preview(self.label_original, crop)
                                        self.update_small_preview(self.label_processed, enhanced_frame[y1:y2, x1:x2])
                                        self.update_small_preview(self.label_contrast, contrast)
                                        self.update_small_preview(self.label_final, final)
                    else:
                        # No detections
                        if frame_count % 30 == 0:  # Log every 30 frames to avoid spam
                            self.log_debug(f"Frame {frame_count}: No detections found")
                        
                        self.result_label.config(text="No money detected")
                        self.confidence_label.config(text="Confidence: --")
                        self.detection_count_label.config(text="Objects detected: 0")
                        self.status_indicator.config(foreground='gray')
                    
                    # Update camera displays
                    self.update_camera_display(self.label_webcam, frame_copy, (640, 480))
                    self.update_camera_display(self.label_enhanced, enhanced_frame, (320, 240))
                    
                except Exception as e:
                    self.log_debug(f"Error in detection loop: {str(e)}")
                    print(f"Detection error: {e}")
                
                # Control frame rate
                time.sleep(0.03)  # ~30 FPS
        
        # Start detection thread
        threading.Thread(target=loop, daemon=True).start()
    
    def update_camera_display(self, label_widget, image, size):
        try:
            img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            img_resized = cv2.resize(img_rgb, size)
            img_tk = ImageTk.PhotoImage(Image.fromarray(img_resized))
            label_widget.imgtk = img_tk
            label_widget.configure(image=img_tk)
        except Exception as e:
            self.log_debug(f"Error updating camera display: {e}")
    
    def update_small_preview(self, label_widget, image):
        try:
            if image is None or image.size == 0:
                return
            
            if len(image.shape) == 2:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            elif len(image.shape) == 3 and image.shape[2] == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            preview = cv2.resize(image, (120, 80))
            img = ImageTk.PhotoImage(Image.fromarray(preview))
            label_widget.imgtk = img
            label_widget.configure(image=img)
        except Exception as e:
            self.log_debug(f"Error updating preview: {e}")
    
    def cleanup(self):
        if hasattr(self, 'cap') and self.cap.isOpened():
            self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    app = MoneyDetectorApp()
    app.root.protocol("WM_DELETE_WINDOW", app.cleanup)  # Ensure cleanup on window close
    app.root.mainloop()