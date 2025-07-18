import os
import sys
import argparse
import glob
import time
import json

import cv2
import numpy as np
import onnxruntime as ort
import RPi.GPIO as GPIO

# Motor control setup
class MotorController:
    def __init__(self, in1=18, in2=19, in3=20, in4=21, ena=12, enb=13):
        self.IN1 = in1
        self.IN2 = in2
        self.IN3 = in3
        self.IN4 = in4
        self.ENA = ena
        self.ENB = enb
        
        # GPIO setup
        GPIO.setmode(GPIO.BCM)
        GPIO.setup([self.IN1, self.IN2, self.IN3, self.IN4], GPIO.OUT)
        GPIO.setup([self.ENA, self.ENB], GPIO.OUT)
        
        # PWM setup for speed control
        self.pwm_a = GPIO.PWM(self.ENA, 1000)
        self.pwm_b = GPIO.PWM(self.ENB, 1000)
        self.pwm_a.start(0)
        self.pwm_b.start(0)
        
        self.speed = 70  # Default speed (0-100)
        
    def move_forward(self, duration=1.0):
        """Move forward for specified duration"""
        GPIO.output(self.IN1, GPIO.HIGH)
        GPIO.output(self.IN2, GPIO.LOW)
        GPIO.output(self.IN3, GPIO.HIGH)
        GPIO.output(self.IN4, GPIO.LOW)
        self.pwm_a.ChangeDutyCycle(self.speed)
        self.pwm_b.ChangeDutyCycle(self.speed)
        time.sleep(duration)
        self.stop()
        
    def turn_left(self, duration=0.5):
        """Turn left for specified duration"""
        GPIO.output(self.IN1, GPIO.LOW)
        GPIO.output(self.IN2, GPIO.HIGH)
        GPIO.output(self.IN3, GPIO.HIGH)
        GPIO.output(self.IN4, GPIO.LOW)
        self.pwm_a.ChangeDutyCycle(self.speed)
        self.pwm_b.ChangeDutyCycle(self.speed)
        time.sleep(duration)
        self.stop()
        
    def turn_right(self, duration=0.5):
        """Turn right for specified duration"""
        GPIO.output(self.IN1, GPIO.HIGH)
        GPIO.output(self.IN2, GPIO.LOW)
        GPIO.output(self.IN3, GPIO.LOW)
        GPIO.output(self.IN4, GPIO.HIGH)
        self.pwm_a.ChangeDutyCycle(self.speed)
        self.pwm_b.ChangeDutyCycle(self.speed)
        time.sleep(duration)
        self.stop()
        
    def stop(self):
        """Stop all motors"""
        GPIO.output([self.IN1, self.IN2, self.IN3, self.IN4], GPIO.LOW)
        self.pwm_a.ChangeDutyCycle(0)
        self.pwm_b.ChangeDutyCycle(0)
        
    def cleanup(self):
        """Clean up GPIO resources"""
        self.stop()
        self.pwm_a.stop()
        self.pwm_b.stop()
        GPIO.cleanup()

# Define and parse user input arguments
parser = argparse.ArgumentParser()
parser.add_argument('--model', help='Path to ONNX model file (example: "yolov5s.onnx")',
                    required=True)
parser.add_argument('--source', help='Image source, can be image file ("test.jpg"), \
                    image folder ("test_dir"), video file ("testvid.mp4"), or index of USB camera ("usb0")', 
                    required=True)
parser.add_argument('--thresh', help='Minimum confidence threshold for displaying detected objects (example: "0.4")',
                    default=0.5, type=float)
parser.add_argument('--resolution', help='Resolution in WxH to display inference results at (example: "640x480"), \
                    otherwise, match source resolution',
                    default=None)
parser.add_argument('--record', help='Record results from video or webcam and save it as "demo1.avi". Must specify --resolution argument to record.',
                    action='store_true')
parser.add_argument('--labels', help='Path to labels file (JSON format with class names)', 
                    default=None)
parser.add_argument('--img-size', help='Input image size for model inference (default: 640)',
                    default=640, type=int)
parser.add_argument('--enable-motors', help='Enable motor control when objects are detected',
                    action='store_true')
parser.add_argument('--motor-speed', help='Motor speed (0-100, default: 70)',
                    default=70, type=int)

args = parser.parse_args()

# Parse user inputs
model_path = args.model
img_source = args.source
min_thresh = args.thresh
user_res = args.resolution
record = args.record
labels_path = args.labels
img_size = args.img_size
enable_motors = args.enable_motors
motor_speed = args.motor_speed

# Initialize motor controller if enabled
motor_controller = None
if enable_motors:
    try:
        motor_controller = MotorController()
        motor_controller.speed = motor_speed
        print(f"Motor controller initialized with speed: {motor_speed}")
    except Exception as e:
        print(f"Warning: Could not initialize motor controller: {e}")
        enable_motors = False

# Check if model file exists and is valid
if not os.path.exists(model_path):
    print('ERROR: Model path is invalid or model was not found. Make sure the model filename was entered correctly.')
    sys.exit(0)

# Load the ONNX model
print(f"Loading ONNX model from {model_path}")
try:
    providers = ['CPUExecutionProvider']
    if 'CUDAExecutionProvider' in ort.get_available_providers():
        providers.insert(0, 'CUDAExecutionProvider')
    
    session = ort.InferenceSession(model_path, providers=providers)
    print(f"Model loaded successfully with providers: {session.get_providers()}")
    
    # Get model input/output info
    input_info = session.get_inputs()[0]
    output_info = session.get_outputs()[0]
    print(f"Model input shape: {input_info.shape}")
    print(f"Model output shape: {output_info.shape}")
    
except Exception as e:
    print(f"ERROR: Failed to load ONNX model: {e}")
    sys.exit(0)

# Load class labels
if labels_path and os.path.exists(labels_path):
    with open(labels_path, 'r') as f:
        labels = json.load(f)
    print(f"Loaded {len(labels)} class labels from {labels_path}")
else:
    # Custom waste detection labels for your model
    labels = ['cans', 'plastic bottle', 'plastic bag', 'water hyacinth', 'water lettuce', 'salvinia']
    print(f"Using custom waste detection labels: {labels}")

# Parse input to determine if image source is a file, folder, video, or USB camera
img_ext_list = ['.jpg','.JPG','.jpeg','.JPEG','.png','.PNG','.bmp','.BMP']
vid_ext_list = ['.avi','.mov','.mp4','.mkv','.wmv']

if os.path.isdir(img_source):
    source_type = 'folder'
elif os.path.isfile(img_source):
    _, ext = os.path.splitext(img_source)
    if ext in img_ext_list:
        source_type = 'image'
    elif ext in vid_ext_list:
        source_type = 'video'
    else:
        print(f'File extension {ext} is not supported.')
        sys.exit(0)
elif 'usb' in img_source:
    source_type = 'usb'
    usb_idx = int(img_source[3:])
elif 'picamera' in img_source:
    source_type = 'picamera'
    picam_idx = int(img_source[8:])
else:
    print(f'Input {img_source} is invalid. Please try again.')
    sys.exit(0)

# Parse user-specified display resolution
resize = False
if user_res:
    resize = True
    resW, resH = int(user_res.split('x')[0]), int(user_res.split('x')[1])

# Check if recording is valid and set up recording
if record:
    if source_type not in ['video','usb']:
        print('Recording only works for video and camera sources. Please try again.')
        sys.exit(0)
    if not user_res:
        print('Please specify resolution to record video at.')
        sys.exit(0)
    
    # Set up recording
    record_name = 'demo1.avi'
    record_fps = 30
    recorder = cv2.VideoWriter(record_name, cv2.VideoWriter_fourcc(*'MJPG'), record_fps, (resW,resH))

# Load or initialize image source
if source_type == 'image':
    imgs_list = [img_source]
elif source_type == 'folder':
    imgs_list = []
    filelist = glob.glob(img_source + '/*')
    for file in filelist:
        _, file_ext = os.path.splitext(file)
        if file_ext in img_ext_list:
            imgs_list.append(file)
elif source_type == 'video' or source_type == 'usb':
    if source_type == 'video': 
        cap_arg = img_source
    elif source_type == 'usb': 
        cap_arg = usb_idx
    cap = cv2.VideoCapture(cap_arg)

    # Set camera or video resolution if specified by user
    if user_res:
        ret = cap.set(3, resW)
        ret = cap.set(4, resH)

elif source_type == 'picamera':
    from picamera2 import Picamera2
    cap = Picamera2()
    cap.configure(cap.create_video_configuration(main={"format": 'XRGB8888', "size": (resW, resH)}))
    cap.start()

# Set bounding box colors for waste detection classes
bbox_colors = [
    (255, 0, 0),     # Red - cans
    (0, 255, 0),     # Green - plastic bottle  
    (0, 0, 255),     # Blue - plastic bag
    (255, 255, 0),   # Yellow - water hyacinth
    (255, 0, 255),   # Magenta - water lettuce
    (0, 255, 255)    # Cyan - salvinia
]

# Initialize control and status variables
avg_frame_rate = 0
frame_rate_buffer = []
fps_avg_len = 200
img_count = 0
last_detection_time = 0
movement_cooldown = 2.0  # Seconds between movements

def preprocess_image(img, target_size):
    """Preprocess image for ONNX model"""
    original_h, original_w = img.shape[:2]
    
    # Calculate scaling factor to maintain aspect ratio
    scale = min(target_size / original_h, target_size / original_w)
    new_h, new_w = int(original_h * scale), int(original_w * scale)
    
    # Resize image
    img_resized = cv2.resize(img, (new_w, new_h))
    
    # Create padded image
    img_padded = np.full((target_size, target_size, 3), 114, dtype=np.uint8)
    
    # Calculate padding offsets
    pad_x = (target_size - new_w) // 2
    pad_y = (target_size - new_h) // 2
    
    # Place resized image in center
    img_padded[pad_y:pad_y+new_h, pad_x:pad_x+new_w] = img_resized
    
    # Convert BGR to RGB and normalize
    img_rgb = cv2.cvtColor(img_padded, cv2.COLOR_BGR2RGB)
    img_normalized = img_rgb.astype(np.float32) / 255.0
    
    # Add batch dimension and convert to NCHW format
    img_tensor = np.transpose(img_normalized, (2, 0, 1))[np.newaxis, ...]
    
    return img_tensor, scale, pad_x, pad_y, original_w, original_h

def postprocess_detections(predictions, original_w, original_h, scale, pad_x, pad_y, conf_threshold=0.5):
    """Post-process ONNX model predictions"""
    detections = []
    
    try:
        # Handle ONNX output format
        if len(predictions.shape) == 3 and predictions.shape[0] == 1:
            predictions = predictions[0]
        
        # Apply confidence threshold and NMS
        for detection in predictions:
            if len(detection) >= 5:
                x_center, y_center, width, height, confidence = detection[:5]
                
                if confidence < conf_threshold:
                    continue
                
                # Get class probabilities if available
                class_id = 0
                if len(detection) > 5:
                    class_scores = detection[5:]
                    class_id = np.argmax(class_scores)
                    class_confidence = class_scores[class_id]
                    confidence *= class_confidence
                
                if confidence < conf_threshold:
                    continue
                
                # Convert coordinates
                x_center = (x_center - pad_x) / scale
                y_center = (y_center - pad_y) / scale
                width = width / scale
                height = height / scale
                
                x1 = max(0, x_center - width / 2)
                y1 = max(0, y_center - height / 2)
                x2 = min(original_w, x_center + width / 2)
                y2 = min(original_h, y_center + height / 2)
                
                if x2 > x1 and y2 > y1:
                    detections.append({
                        'bbox': [int(x1), int(y1), int(x2), int(y2)],
                        'confidence': confidence,
                        'class_id': min(class_id, len(labels) - 1),
                        'class_name': labels[min(class_id, len(labels) - 1)]
                    })
    
    except Exception as e:
        print(f"Error in postprocessing: {e}")
    
    return detections

def control_movement(detections, frame_width):
    """Control robot movement based on detections"""
    global last_detection_time, motor_controller
    
    if not enable_motors or not motor_controller:
        return
        
    current_time = time.time()
    
    # Check cooldown period
    if current_time - last_detection_time < movement_cooldown:
        return
    
    if len(detections) == 0:
        return
    
    # Find the largest detection (closest object)
    largest_detection = max(detections, key=lambda d: (d['bbox'][2] - d['bbox'][0]) * (d['bbox'][3] - d['bbox'][1]))
    
    # Get center of detection
    x1, y1, x2, y2 = largest_detection['bbox']
    center_x = (x1 + x2) / 2
    frame_center = frame_width / 2
    
    print(f"Detected {largest_detection['class_name']} at center: {center_x:.0f}")
    
    # Movement logic based on object position
    if center_x < frame_center - 50:  # Object on left
        print("Turning left towards object")
        motor_controller.turn_left(0.3)
    elif center_x > frame_center + 50:  # Object on right
        print("Turning right towards object")
        motor_controller.turn_right(0.3)
    else:  # Object in center
        print("Moving forward towards object")
        motor_controller.move_forward(0.5)
    
    last_detection_time = current_time

# Begin inference loop
print("Starting inference loop...")
frame_count = 0

try:
    while True:
        t_start = time.perf_counter()
        frame_count += 1

        # Load frame from image source
        if source_type == 'image' or source_type == 'folder':
            if img_count >= len(imgs_list):
                print('All images have been processed. Exiting program.')
                break
            img_filename = imgs_list[img_count]
            frame = cv2.imread(img_filename)
            if frame is None:
                print(f'Failed to load image: {img_filename}')
                img_count += 1
                continue
            img_count += 1
        
        elif source_type == 'video':
            ret, frame = cap.read()
            if not ret:
                print('Reached end of the video file. Exiting program.')
                break
        
        elif source_type == 'usb':
            ret, frame = cap.read()
            if not ret or frame is None:
                print('Unable to read frames from the camera. Exiting program.')
                break

        elif source_type == 'picamera':
            try:
                frame_bgra = cap.capture_array()
                frame = cv2.cvtColor(np.copy(frame_bgra), cv2.COLOR_BGRA2BGR)
                if frame is None:
                    print('Unable to read frames from the Picamera. Exiting program.')
                    break
            except Exception as e:
                print(f'Picamera error: {e}')
                break

        # Store original frame for display
        display_frame = frame.copy()
        original_shape = frame.shape

        # Resize frame to desired display resolution
        if resize:
            display_frame = cv2.resize(display_frame, (resW, resH))

        # Preprocess image for model inference
        try:
            input_tensor, scale, pad_x, pad_y, original_w, original_h = preprocess_image(frame, img_size)
        except Exception as e:
            print(f"Error in preprocessing: {e}")
            continue

        # Run ONNX inference
        try:
            input_name = session.get_inputs()[0].name
            predictions = session.run(None, {input_name: input_tensor})[0]
        except Exception as e:
            print(f"ONNX inference error: {e}")
            continue

        # Post-process predictions
        try:
            detections = postprocess_detections(predictions, original_w, original_h, scale, pad_x, pad_y, min_thresh)
        except Exception as e:
            print(f"Error in postprocessing: {e}")
            detections = []

        # Control robot movement based on detections
        if source_type in ['usb', 'picamera']:  # Only for live camera feeds
            control_movement(detections, original_w)

        # Object counting
        object_count = len(detections)

        # Draw detections on display frame
        for detection in detections:
            x1, y1, x2, y2 = detection['bbox']
            confidence = detection['confidence']
            class_name = detection['class_name']
            class_id = detection['class_id']

            # Scale coordinates if display frame is resized
            if resize:
                scale_x = resW / original_w
                scale_y = resH / original_h
                x1 = int(x1 * scale_x)
                y1 = int(y1 * scale_y)
                x2 = int(x2 * scale_x)
                y2 = int(y2 * scale_y)

            # Draw bounding box
            color = bbox_colors[class_id % len(bbox_colors)]
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)

            # Draw label
            label = f'{class_name}: {confidence:.2f}'
            labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            label_ymin = max(y1, labelSize[1] + 10)
            cv2.rectangle(display_frame, (x1, label_ymin-labelSize[1]-10), 
                         (x1+labelSize[0], label_ymin+baseLine-10), color, cv2.FILLED)
            cv2.putText(display_frame, label, (x1, label_ymin-7), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        # Display motor status
        if enable_motors:
            motor_status = "Motors: ENABLED" if motor_controller else "Motors: ERROR"
            cv2.putText(display_frame, motor_status, (10, 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0) if motor_controller else (0, 0, 255), 2)

        # Calculate and draw framerate
        if source_type in ['video', 'usb', 'picamera']:
            cv2.putText(display_frame, f'FPS: {avg_frame_rate:.1f}', (10, 40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # Display detection count
        cv2.putText(display_frame, f'Objects: {object_count}', (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # Display the frame
        cv2.imshow('Waste Detection with Motor Control', display_frame)
        
        if record: 
            recorder.write(display_frame)

        # Handle key presses
        if source_type in ['image', 'folder']:
            key = cv2.waitKey(0)
        elif source_type in ['video', 'usb', 'picamera']:
            key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q') or key == ord('Q'):
            break
        elif key == ord('s') or key == ord('S'):
            if motor_controller:
                motor_controller.stop()
            print("Stopped. Press any key to continue...")
            cv2.waitKey(0)
        
        # Calculate FPS
        t_stop = time.perf_counter()
        frame_rate_calc = 1.0 / (t_stop - t_start)

        # Update frame rate buffer
        if len(frame_rate_buffer) >= fps_avg_len:
            frame_rate_buffer.pop(0)
        frame_rate_buffer.append(frame_rate_calc)

        # Calculate average FPS
        avg_frame_rate = np.mean(frame_rate_buffer)

except KeyboardInterrupt:
    print("\nInterrupted by user")

finally:
    # Cleanup
    print(f'Processing complete. Average FPS: {avg_frame_rate:.2f}')
    if source_type in ['video', 'usb']:
        cap.release()
    elif source_type == 'picamera':
        cap.stop()
    if record: 
        recorder.release()
    if motor_controller:
        motor_controller.cleanup()
    cv2.destroyAllWindows()
