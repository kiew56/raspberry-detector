import os
import sys
import argparse
import glob
import time
import json

import cv2
import numpy as np
import torch

# Define and parse user input arguments
parser = argparse.ArgumentParser()
parser.add_argument('--model', help='Path to YOLOv5 TorchScript model file (example: "yolov5s.torchscript")',
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

args = parser.parse_args()

# Parse user inputs
model_path = args.model
img_source = args.source
min_thresh = args.thresh
user_res = args.resolution
record = args.record
labels_path = args.labels
img_size = args.img_size

# Check if model file exists and is valid
if not os.path.exists(model_path):
    print('ERROR: Model path is invalid or model was not found. Make sure the model filename was entered correctly.')
    sys.exit(0)

# Load the TorchScript model
print(f"Loading TorchScript model from {model_path}")
try:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = torch.jit.load(model_path, map_location=device)
    model.eval()
    print(f"Model loaded successfully on {device}")
except Exception as e:
    print(f"ERROR: Failed to load TorchScript model: {e}")
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
# Colors chosen for good visibility and distinction
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
    Preprocess image for YOLOv5 TorchScript model
    """
    # Resize image while maintaining aspect ratio
    h, w = img.shape[:2]
    scale = min(target_size / h, target_size / w)
    new_h, new_w = int(h * scale), int(w * scale)
    
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
    
    # Convert to tensor format (CHW)
    img_tensor = torch.from_numpy(img_normalized).permute(2, 0, 1).unsqueeze(0).to(device)
    
    return img_tensor, scale, pad_x, pad_y

def postprocess_detections(predictions, original_shape, scale, pad_x, pad_y, conf_threshold=0.5):
    """
    Post-process YOLOv5 TorchScript model predictions
    """
    # predictions shape: [batch_size, num_detections, 85] (for COCO)
    # Each detection: [x_center, y_center, width, height, confidence, class_scores...]
    
    detections = []
    
    if len(predictions.shape) == 3:
        predictions = predictions[0]  # Remove batch dimension
    
    for detection in predictions:
        # Extract confidence and class scores
        confidence = detection[4].item()
        
        if confidence > conf_threshold:
            # Get class with highest score
            class_scores = detection[5:]
            class_id = torch.argmax(class_scores).item()
            class_confidence = class_scores[class_id].item()
            
            # Calculate final confidence
            final_confidence = confidence * class_confidence
            
            if final_confidence > conf_threshold:
                # Convert from center coordinates to corner coordinates
                x_center, y_center, width, height = detection[:4]
                
                # Convert from normalized coordinates to pixel coordinates
                x_center = x_center.item()
                y_center = y_center.item()
                width = width.item()
                height = height.item()
                
                # Adjust for padding
                x_center = (x_center - pad_x) / scale
                y_center = (y_center - pad_y) / scale
                width = width / scale
                height = height / scale
                
                # Convert to corner coordinates
                x1 = int(x_center - width / 2)
                y1 = int(y_center - height / 2)
                x2 = int(x_center + width / 2)
                y2 = int(y_center + height / 2)
                
                # Clip to image boundaries
                x1 = max(0, min(x1, original_shape[1]))
                y1 = max(0, min(y1, original_shape[0]))
                x2 = max(0, min(x2, original_shape[1]))
                y2 = max(0, min(y2, original_shape[0]))
                
                detections.append({
                    'bbox': [x1, y1, x2, y2],
                    'confidence': final_confidence,
                    'class_id': class_id,
                    'class_name': labels[class_id] if class_id < len(labels) else f'class_{class_id}'
                })
    
    return detections

# Begin inference loop
print("Starting inference loop...")
while True:
    t_start = time.perf_counter()

    # Load frame from image source
    if source_type == 'image' or source_type == 'folder':
        if img_count >= len(imgs_list):
            print('All images have been processed. Exiting program.')
            sys.exit(0)
        img_filename = imgs_list[img_count]
        frame = cv2.imread(img_filename)
        img_count = img_count + 1
    
    elif source_type == 'video':
        ret, frame = cap.read()
        if not ret:
            print('Reached end of the video file. Exiting program.')
            break
    
    elif source_type == 'usb':
        ret, frame = cap.read()
        if (frame is None) or (not ret):
            print('Unable to read frames from the camera. This indicates the camera is disconnected or not working. Exiting program.')
            break

    elif source_type == 'picamera':
        frame_bgra = cap.capture_array()
        frame = cv2.cvtColor(np.copy(frame_bgra), cv2.COLOR_BGRA2BGR)
        if (frame is None):
            print('Unable to read frames from the Picamera. This indicates the camera is disconnected or not working. Exiting program.')
            break

    # Store original frame for display
    display_frame = frame.copy()
    original_shape = frame.shape

    # Resize frame to desired display resolution
    if resize == True:
        display_frame = cv2.resize(display_frame, (resW, resH))

    # Preprocess image for model inference
    input_tensor, scale, pad_x, pad_y = preprocess_image(frame, img_size)

    # Run inference
    with torch.no_grad():
        predictions = model(input_tensor)

    # Post-process predictions
    detections = postprocess_detections(predictions, original_shape, scale, pad_x, pad_y, min_thresh)

    # Initialize variable for basic object counting example
    object_count = len(detections)

    # Draw detections on display frame
    for detection in detections:
        x1, y1, x2, y2 = detection['bbox']
        confidence = detection['confidence']
        class_name = detection['class_name']
        class_id = detection['class_id']

        # Scale coordinates if display frame is resized
        if resize:
            scale_x = resW / original_shape[1]
            scale_y = resH / original_shape[0]
            x1 = int(x1 * scale_x)
            y1 = int(y1 * scale_y)
            x2 = int(x2 * scale_x)
            y2 = int(y2 * scale_y)

        # Draw bounding box
        color = bbox_colors[class_id % len(bbox_colors)]
        cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)

        # Draw label
        label = f'{class_name}: {int(confidence*100)}%'
        labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        label_ymin = max(y1, labelSize[1] + 10)
        cv2.rectangle(display_frame, (x1, label_ymin-labelSize[1]-10), 
                     (x1+labelSize[0], label_ymin+baseLine-10), color, cv2.FILLED)
        cv2.putText(display_frame, label, (x1, label_ymin-7), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    # Calculate and draw framerate (if using video, USB, or Picamera source)
    if source_type in ['video', 'usb', 'picamera']:
        cv2.putText(display_frame, f'FPS: {avg_frame_rate:0.2f}', (10,20), 
                   cv2.FONT_HERSHEY_SIMPLEX, .7, (0,255,255), 2)
    
    # Display detection results with waste category counts
    waste_counts = {}
    for detection in detections:
        class_name = detection['class_name']
        waste_counts[class_name] = waste_counts.get(class_name, 0) + 1
    
    # Display total count
    cv2.putText(display_frame, f'Total waste items: {object_count}', (10,40), 
               cv2.FONT_HERSHEY_SIMPLEX, .7, (0,255,255), 2)
    
    # Display individual waste type counts
    y_offset = 60
    for waste_type, count in waste_counts.items():
        cv2.putText(display_frame, f'{waste_type}: {count}', (10, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, .5, (0,255,255), 1)
        y_offset += 20
    
    cv2.imshow('Waste Detection - YOLOv5 TorchScript', display_frame)
    
    if record: 
        recorder.write(display_frame)

    # Handle key presses
    if source_type in ['image', 'folder']:
        key = cv2.waitKey()
    elif source_type in ['video', 'usb', 'picamera']:
        key = cv2.waitKey(5)
    
    if key == ord('q') or key == ord('Q'):  # Press 'q' to quit
        break
    elif key == ord('s') or key == ord('S'):  # Press 's' to pause inference
        cv2.waitKey()
    elif key == ord('p') or key == ord('P'):  # Press 'p' to save a picture of results on this frame
        cv2.imwrite('capture.png', display_frame)
    
    # Calculate FPS for this frame
    t_stop = time.perf_counter()
    frame_rate_calc = float(1/(t_stop - t_start))

    # Append FPS result to frame_rate_buffer (for finding average FPS over multiple frames)
    if len(frame_rate_buffer) >= fps_avg_len:
        frame_rate_buffer.pop(0)
        frame_rate_buffer.append(frame_rate_calc)
    else:
        frame_rate_buffer.append(frame_rate_calc)

    # Calculate average FPS for past frames
    avg_frame_rate = np.mean(frame_rate_buffer)

# Clean up
print(f'Average pipeline FPS: {avg_frame_rate:.2f}')
if source_type in ['video', 'usb']:
    cap.release()
elif source_type == 'picamera':
    cap.stop()
if record: 
    recorder.release()
cv2.destroyAllWindows()
