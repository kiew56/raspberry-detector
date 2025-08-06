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

# Define and parse user input arguments
parser = argparse.ArgumentParser()
#parser.add_argument('--model', help='Path to YOLOv5 ONNX model file (example: "yolov5s.onnx")',
 #                   required=True)
#parser.add_argument('--source', help='Image source, can be image file ("test.jpg"), \
#                    image folder ("test_dir"), video file ("testvid.mp4"), or index of USB camera ("usb0")', 
#                    required=True)
parser.add_argument('--thresh', help='Minimum confidence threshold for displaying detected objects (example: "0.4")',
                   default=0.5, type=float)
#parser.add_argument('--resolution', help='Resolution in WxH to display inference results at (example: "640x480"), \
 #                   otherwise, match source resolution',
 #                   default=None)
parser.add_argument('--record', help='Record results from video or webcam and save it as "demo1.avi". Must specify --resolution argument to record.',
                    action='store_true')
parser.add_argument('--labels', help='Path to labels file (JSON format with class names)', 
                    default=None)
parser.add_argument('--img-size', help='Input image size for model inference (default: 640)',
                    default=640, type=int)

args = parser.parse_args()

# Parse user inputs
model_path = 'best.onnx'
img_source = 'picamera0'
min_thresh = args.thresh
user_res = '480x360'
record = args.record
labels_path = args.labels
img_size = args.img_size

# Check if model file exists and is valid
if not os.path.exists(model_path):
    print('ERROR: Model path is invalid or model was not found. Make sure the model filename was entered correctly.')
    sys.exit(0)

# Load the ONNX model
print(f"Loading ONNX model from {model_path}")
try:
    session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
    print(f"Model loaded successfully with provider: CPUExecutionProvider")
except Exception as e:
    print(f"ERROR: Failed to load ONNX model: {e}")
    sys.exit(0)

# Load class labels
if labels_path and os.path.exists(labels_path):
    with open(labels_path, 'r') as f:
        labels = json.load(f)
    print(f"Loaded {len(labels)} class labels from {labels_path}")
else:
    labels = ['water lettuce', 'water hyacinth', 'plastic bottle', 'salvinia', 'cans', 'plastic bag']
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
    if source_type not in ['video','usb','picamera']:
        print('Recording only works for video and camera sources. Please try again.')
        sys.exit(0)
    if not user_res:
        print('Please specify resolution to record video at.')
        sys.exit(0)
    
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
    if user_res:
        cap.set(3, resW)
        cap.set(4, resH)
elif source_type == 'picamera':
    from picamera2 import Picamera2
    cap = Picamera2()
    cap.configure(cap.create_video_configuration(main={"format": 'XRGB8888', "size": (resW, resH)}))
    cap.start()

# Set up L298N motor driver GPIO pins
GPIO.setmode(GPIO.BCM)
ENA = 25  # Enable A (Motor A speed)
IN1 = 23  # Motor A input 1
IN2 = 24  # Motor A input 2
ENB = 17  # Enable B (Motor B speed)
IN3 = 27  # Motor B input 1
IN4 = 22  # Motor B input 2
GPIO.setup(ENA, GPIO.OUT)
GPIO.setup(IN1, GPIO.OUT)
GPIO.setup(IN2, GPIO.OUT)
GPIO.setup(ENB, GPIO.OUT)
GPIO.setup(IN3, GPIO.OUT)
GPIO.setup(IN4, GPIO.OUT)

# Initialize PWM for speed control
pwm_a = GPIO.PWM(ENA, 1000)  # 1000 Hz frequency
pwm_b = GPIO.PWM(ENB, 1000)
pwm_a.start(0)  # Start PWM with 0% duty cycle
pwm_b.start(0)

# Motor control functions
def motor_forward(speed=50):
    GPIO.output(IN1, GPIO.HIGH)
    GPIO.output(IN2, GPIO.LOW)
    GPIO.output(IN3, GPIO.HIGH)
    GPIO.output(IN4, GPIO.LOW)
    pwm_a.ChangeDutyCycle(speed)
    pwm_b.ChangeDutyCycle(speed)

def motor_backward(speed=50):
    GPIO.output(IN1, GPIO.LOW)
    GPIO.output(IN2, GPIO.HIGH)
    GPIO.output(IN3, GPIO.LOW)
    GPIO.output(IN4, GPIO.HIGH)
    pwm_a.ChangeDutyCycle(speed)
    pwm_b.ChangeDutyCycle(speed)

def motor_left(speed=50):
    GPIO.output(IN1, GPIO.LOW)
    GPIO.output(IN2, GPIO.HIGH)
    GPIO.output(IN3, GPIO.HIGH)
    GPIO.output(IN4, GPIO.LOW)
    pwm_a.ChangeDutyCycle(speed)
    pwm_b.ChangeDutyCycle(speed)

def motor_right(speed=50):
    GPIO.output(IN1, GPIO.HIGH)
    GPIO.output(IN2, GPIO.LOW)
    GPIO.output(IN3, GPIO.LOW)
    GPIO.output(IN4, GPIO.HIGH)
    pwm_a.ChangeDutyCycle(speed)
    pwm_b.ChangeDutyCycle(speed)

def motor_stop():
    GPIO.output(IN1, GPIO.LOW)
    GPIO.output(IN2, GPIO.LOW)
    GPIO.output(IN3, GPIO.LOW)
    GPIO.output(IN4, GPIO.LOW)
    pwm_a.ChangeDutyCycle(0)
    pwm_b.ChangeDutyCycle(0)

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

def preprocess_image(img, target_size):
    """
    Preprocess image for YOLOv5 ONNX model
    """
    original_h, original_w = img.shape[:2]
    scale = min(target_size / original_h, target_size / original_w)
    new_h, new_w = int(original_h * scale), int(original_w * scale)
    img_resized = cv2.resize(img, (new_w, new_h))
    img_padded = np.full((target_size, target_size, 3), 114, dtype=np.uint8)
    pad_x = (target_size - new_w) // 2
    pad_y = (target_size - new_h) // 2
    img_padded[pad_y:pad_y+new_h, pad_x:pad_x+new_w] = img_resized
    img_rgb = cv2.cvtColor(img_padded, cv2.COLOR_BGR2RGB)
    img_normalized = img_rgb.astype(np.float32) / 255.0
    img_tensor = np.transpose(img_normalized, (2, 0, 1))
    img_tensor = np.expand_dims(img_tensor, axis=0)
    return img_tensor, scale, pad_x, pad_y, original_w, original_h

def postprocess_detections(predictions, original_w, original_h, scale, pad_x, pad_y, conf_threshold=0.5):
    """
    Post-process YOLOv5 ONNX model predictions with flexible handling
    """
    detections = []
    if isinstance(predictions, (list, tuple)):
        predictions = predictions[0]
    if len(predictions.shape) == 3:
        predictions = predictions[0]
    num_classes = len(labels)
    expected_channels = 5 + num_classes
    print(f"Prediction shape: {predictions.shape}")
    print(f"Expected channels: {expected_channels}, Actual channels: {predictions.shape[-1]}")
    for detection in predictions:
        if len(detection) < 5:
            continue
        x_center, y_center, width, height, confidence = detection[:5]
        if confidence < conf_threshold:
            continue
        if len(detection) > 5:
            class_scores = detection[5:5+num_classes] if len(detection) >= 5+num_classes else detection[5:]
            if len(class_scores) > 0:
                class_id = np.argmax(class_scores)
                class_confidence = class_scores[class_id]
            else:
                class_id = 0
                class_confidence = 0.5
        else:
            class_id = 0
            class_confidence = confidence
        class_id = min(class_id, len(labels) - 1)
        final_confidence = confidence * class_confidence
        if final_confidence > conf_threshold:
            x_center = float(x_center)
            y_center = float(y_center)
            width = float(width)
            height = float(height)
            x_center = (x_center - pad_x) / scale
            y_center = (y_center - pad_y) / scale
            width = width / scale
            height = height / scale
            x1 = int(x_center - width / 2)
            y1 = int(y_center - height / 2)
            x2 = int(x_center + width / 2)
            y2 = int(y_center + height / 2)
            x1 = max(0, min(x1, original_w))
            y1 = max(0, min(y1, original_h))
            x2 = max(0, min(x2, original_w))
            y2 = max(0, min(y2, original_h))
            if x2 <= x1 or y2 <= y1:
                continue
            detections.append({
                'bbox': [x1, y1, x2, y2],
                'confidence': final_confidence,
                'class_id': class_id,
                'class_name': labels[class_id]
            })
    return detections

# Navigation logic
def navigate_to_object(detections, frame_width):
    """
    Determine motor commands based on detected object's position
    """
    target_classes = ['water lettuce', 'water hyacinth', 'plastic bottle', 'salvinia', 'cans', 'plastic bag']
    target_detection = None
    max_area = 0
    for detection in detections:
        if detection['class_name'] in target_classes:
            x1, y1, x2, y2 = detection['bbox']
            area = (x2 - x1) * (y2 - y1)
            if area > max_area:
                max_area = area
                target_detection = detection
    if target_detection:
        x1, y1, x2, y2 = target_detection['bbox']
        obj_center = (x1 + x2) // 2
        frame_center = frame_width // 2
        threshold = frame_width // 4  # Adjust based on desired sensitivity
        if obj_center < frame_center - threshold:
            print(f"Object ({target_detection['class_name']}) on left, turning left")
            motor_left(speed=50)
        elif obj_center > frame_center + threshold:
            print(f"Object ({target_detection['class_name']}) on right, turning right")
            motor_right(speed=50)
        else:
            print(f"Object ({target_detection['class_name']}) centered, moving forward")
            motor_forward(speed=50)
        return True
    else:
        print("No target object detected, stopping")
        motor_stop()
        return False

# Begin inference loop
print("Starting inference loop...")
try:
    while True:
        t_start = time.perf_counter()
        if source_type == 'image' or source_type == 'folder':
            if img_count >= len(imgs_list):
                print('All images have been processed. Exiting program.')
                break
            img_filename = imgs_list[img_count]
            frame = cv2.imread(img_filename)
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
            frame_bgra = cap.capture_array()
            frame = cv2.cvtColor(np.copy(frame_bgra), cv2.COLOR_BGRA2BGR)
            if frame is None:
                print('Unable to read frames from the Picamera. Exiting program.')
                break
        display_frame = frame.copy()
        original_shape = frame.shape
        if resize:
            display_frame = cv2.resize(display_frame, (resW, resH))
        try:
            input_tensor, scale, pad_x, pad_y, original_w, original_h = preprocess_image(frame, img_size)
        except Exception as e:
            print(f"Error in preprocessing: {e}")
            continue
        try:
            input_name = session.get_inputs()[0].name
            predictions = session.run(None, {input_name: input_tensor})[0]
        except Exception as e:
            print(f"Model inference error: {e}")
            break
        try:
            detections = postprocess_detections(predictions, original_w, original_h, scale, pad_x, pad_y, min_thresh)
        except Exception as e:
            print(f"Error in postprocessing: {e}")
            detections = []
        navigate_to_object(detections, original_w)
        object_count = len(detections)
        for detection in detections:
            x1, y1, x2, y2 = detection['bbox']
            confidence = detection['confidence']
            class_name = detection['class_name']
            class_id = detection['class_id']
            if resize:
                scale_x = resW / original_w
                scale_y = resH / original_h
                x1 = int(x1 * scale_x)
                y1 = int(y1 * scale_y)
                x2 = int(x2 * scale_x)
                y2 = int(y2 * scale_y)
            color = bbox_colors[class_id % len(bbox_colors)]
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
            label = f'{class_name}: {int(confidence*100)}%'
            labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            label_ymin = max(y1, labelSize[1] + 10)
            cv2.rectangle(display_frame, (x1, label_ymin-labelSize[1]-10), 
                         (x1+labelSize[0], label_ymin+baseLine-10), color, cv2.FILLED)
            cv2.putText(display_frame, label, (x1, label_ymin-7), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        if source_type in ['video', 'usb', 'picamera']:
            cv2.putText(display_frame, f'FPS: {avg_frame_rate:0.2f}', (10,20), 
                       cv2.FONT_HERSHEY_SIMPLEX, .7, (0,255,255), 2)
        waste_counts = {}
        for detection in detections:
            class_name = detection['class_name']
            waste_counts[class_name] = waste_counts.get(class_name, 0) + 1
        cv2.putText(display_frame, f'Total waste items: {object_count}', (10,40), 
                   cv2.FONT_HERSHEY_SIMPLEX, .7, (0,255,255), 2)
        y_offset = 60
        for waste_type, count in waste_counts.items():
            cv2.putText(display_frame, f'{waste_type}: {count}', (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, .5, (0,255,255), 1)
            y_offset += 20
        cv2.imshow('Waste Detection - YOLOv5 ONNX', display_frame)
        if record: 
            recorder.write(display_frame)
        if source_type in ['image', 'folder']:
            key = cv2.waitKey(0)
        else:
            key = cv2.waitKey(5)
        if key == ord('q') or key == ord('Q'):
            break
        elif key == ord('s') or key == ord('S'):
            cv2.waitKey(0)
        elif key == ord('p') or key == ord('P'):
            cv2.imwrite('capture.png', display_frame)
        t_stop = time.perf_counter()
        frame_rate_calc = float(1/(t_stop - t_start))
        if len(frame_rate_buffer) >= fps_avg_len:
            frame_rate_buffer.pop(0)
            frame_rate_buffer.append(frame_rate_calc)
        else:
            frame_rate_buffer.append(frame_rate_calc)
        avg_frame_rate = np.mean(frame_rate_buffer)
finally:
    print(f'Average pipeline FPS: {avg_frame_rate:.2f}')
    if source_type in ['video', 'usb']:
        cap.release()
    elif source_type == 'picamera':
        cap.stop()
    if record: 
        recorder.release()
    motor_stop()
    GPIO.cleanup()
    cv2.destroyAllWindows()
