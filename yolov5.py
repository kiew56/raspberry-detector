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
parser.add_argument('--model-info', help='Debug model information', action='store_true')

args = parser.parse_args()

# Parse user inputs
model_path = args.model
img_source = args.source
min_thresh = args.thresh
user_res = args.resolution
record = args.record
labels_path = args.labels
img_size = args.img_size
debug_model = args.model_info

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
    
    # Debug model information
    if debug_model:
        print("\n=== MODEL DEBUG INFO ===")
        try:
            # Try to get model info
            test_input = torch.randn(1, 3, img_size, img_size).to(device)
            with torch.no_grad():
                test_output = model(test_input)
            print(f"Model input shape: {test_input.shape}")
            print(f"Model output type: {type(test_output)}")
            if isinstance(test_output, (list, tuple)):
                print(f"Model output length: {len(test_output)}")
                for i, out in enumerate(test_output):
                    print(f"Output {i} shape: {out.shape}")
            else:
                print(f"Model output shape: {test_output.shape}")
        except Exception as e:
            print(f"Model debug failed: {e}")
        print("========================\n")
    
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
    Preprocess image for YOLOv5 TorchScript model with proper normalization
    """
    original_h, original_w = img.shape[:2]
    
    # Calculate scaling factor to maintain aspect ratio
    scale = min(target_size / original_h, target_size / original_w)
    new_h, new_w = int(original_h * scale), int(original_w * scale)
    
    # Resize image
    img_resized = cv2.resize(img, (new_w, new_h))
    
    # Create padded image with gray padding (114 is common in YOLO)
    img_padded = np.full((target_size, target_size, 3), 114, dtype=np.uint8)
    
    # Calculate padding offsets to center the image
    pad_x = (target_size - new_w) // 2
    pad_y = (target_size - new_h) // 2
    
    # Place resized image in center of padded image
    img_padded[pad_y:pad_y+new_h, pad_x:pad_x+new_w] = img_resized
    
    # Convert BGR to RGB (YOLOv5 expects RGB)
    img_rgb = cv2.cvtColor(img_padded, cv2.COLOR_BGR2RGB)
    
    # Normalize to [0, 1] range
    img_normalized = img_rgb.astype(np.float32) / 255.0
    
    # Convert to tensor format (CHW) and add batch dimension
    img_tensor = torch.from_numpy(img_normalized).permute(2, 0, 1).unsqueeze(0).to(device)
    
    return img_tensor, scale, pad_x, pad_y, original_w, original_h

def apply_nms(boxes, scores, classes, iou_threshold=0.5):
    """
    Apply Non-Maximum Suppression to filter overlapping detections
    """
    if len(boxes) == 0:
        return []
    
    # Convert to numpy for easier processing
    boxes = np.array(boxes)
    scores = np.array(scores)
    classes = np.array(classes)
    
    # Calculate areas
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    
    # Sort by confidence scores (descending)
    indices = np.argsort(scores)[::-1]
    
    keep = []
    while len(indices) > 0:
        # Pick the detection with highest confidence
        current = indices[0]
        keep.append(current)
        
        if len(indices) == 1:
            break
            
        # Calculate IoU with remaining detections
        current_box = boxes[current]
        remaining_boxes = boxes[indices[1:]]
        
        # Calculate intersection
        x1 = np.maximum(current_box[0], remaining_boxes[:, 0])
        y1 = np.maximum(current_box[1], remaining_boxes[:, 1])
        x2 = np.minimum(current_box[2], remaining_boxes[:, 2])
        y2 = np.minimum(current_box[3], remaining_boxes[:, 3])
        
        intersection = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
        
        # Calculate union
        current_area = areas[current]
        remaining_areas = areas[indices[1:]]
        union = current_area + remaining_areas - intersection
        
        # Calculate IoU
        iou = intersection / (union + 1e-6)
        
        # Keep detections with IoU below threshold
        indices = indices[1:][iou <= iou_threshold]
    
    return keep

def postprocess_detections(predictions, original_w, original_h, scale, pad_x, pad_y, conf_threshold=0.5, iou_threshold=0.5):
    """
    Enhanced post-processing for YOLOv5 TorchScript predictions with better error handling
    """
    detections = []
    
    try:
        # Handle different prediction formats
        if isinstance(predictions, (list, tuple)):
            # Some models return multiple outputs, typically we want the first one
            predictions = predictions[0]
        
        # Remove batch dimension if present
        if len(predictions.shape) == 3 and predictions.shape[0] == 1:
            predictions = predictions[0]
        
        print(f"Processing predictions with shape: {predictions.shape}")
        
        # Handle different output formats
        if len(predictions.shape) == 2:
            # Standard format: [num_detections, prediction_size]
            num_detections, prediction_size = predictions.shape
            print(f"Standard format: {num_detections} detections, {prediction_size} values per detection")
            
            # Calculate expected format
            num_classes = len(labels)
            expected_size = 5 + num_classes  # x, y, w, h, obj_conf + class_probs
            
            if prediction_size < 5:
                print(f"Warning: Insufficient prediction data ({prediction_size} < 5)")
                return detections
            
            # Process each detection
            boxes, scores, classes = [], [], []
            
            for i in range(num_detections):
                detection = predictions[i]
                
                # Extract basic detection info
                if len(detection) >= 5:
                    x_center = float(detection[0])
                    y_center = float(detection[1])
                    width = float(detection[2])
                    height = float(detection[3])
                    obj_conf = float(detection[4])
                    
                    # Skip low confidence detections early
                    if obj_conf < conf_threshold:
                        continue
                    
                    # Handle class predictions
                    if prediction_size > 5:
                        # Extract class probabilities
                        class_probs = detection[5:5+num_classes] if prediction_size >= 5+num_classes else detection[5:]
                        
                        if len(class_probs) > 0:
                            # Find class with highest probability
                            if isinstance(class_probs, torch.Tensor):
                                class_probs = class_probs.cpu().numpy()
                            
                            class_id = np.argmax(class_probs)
                            class_prob = float(class_probs[class_id]) if class_id < len(class_probs) else 0.5
                        else:
                            class_id = 0
                            class_prob = 1.0
                    else:
                        # No class probabilities, assume single class
                        class_id = 0
                        class_prob = 1.0
                    
                    # Calculate final confidence
                    final_conf = obj_conf * class_prob
                    
                    if final_conf < conf_threshold:
                        continue
                    
                    # Ensure class_id is valid
                    class_id = min(max(0, int(class_id)), len(labels) - 1)
                    
                    # Convert from center format to corner format
                    # Adjust for padding and scaling
                    x_center = (x_center - pad_x) / scale
                    y_center = (y_center - pad_y) / scale
                    width = width / scale
                    height = height / scale
                    
                    # Convert to corner coordinates
                    x1 = x_center - width / 2
                    y1 = y_center - height / 2
                    x2 = x_center + width / 2
                    y2 = y_center + height / 2
                    
                    # Clip to image boundaries
                    x1 = max(0, min(x1, original_w))
                    y1 = max(0, min(y1, original_h))
                    x2 = max(0, min(x2, original_w))
                    y2 = max(0, min(y2, original_h))
                    
                    # Skip invalid boxes
                    if x2 <= x1 or y2 <= y1:
                        continue
                    
                    boxes.append([x1, y1, x2, y2])
                    scores.append(final_conf)
                    classes.append(class_id)
            
            # Apply Non-Maximum Suppression
            if len(boxes) > 0:
                keep_indices = apply_nms(boxes, scores, classes, iou_threshold)
                
                for idx in keep_indices:
                    detections.append({
                        'bbox': [int(coord) for coord in boxes[idx]],
                        'confidence': scores[idx],
                        'class_id': classes[idx],
                        'class_name': labels[classes[idx]]
                    })
        
        else:
            print(f"Unexpected prediction format with shape: {predictions.shape}")
            # Try to handle as flattened format
            if len(predictions.shape) == 1:
                print("Attempting to handle flattened prediction format")
                # This would require knowing the exact model output format
                # For now, return empty detections
                return detections
    
    except Exception as e:
        print(f"Error in postprocessing: {e}")
        print(f"Predictions type: {type(predictions)}")
        if hasattr(predictions, 'shape'):
            print(f"Predictions shape: {predictions.shape}")
        return detections
    
    return detections

# Begin inference loop
print("Starting inference loop...")
frame_count = 0

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

    # Run inference with comprehensive error handling
    try:
        with torch.no_grad():
            predictions = model(input_tensor)
            
        # Debug first few frames
        if frame_count <= 3:
            print(f"Frame {frame_count} - Prediction type: {type(predictions)}")
            if isinstance(predictions, (list, tuple)):
                print(f"Prediction is list/tuple with {len(predictions)} elements")
                for i, pred in enumerate(predictions):
                    if hasattr(pred, 'shape'):
                        print(f"  Element {i} shape: {pred.shape}")
            elif hasattr(predictions, 'shape'):
                print(f"Prediction shape: {predictions.shape}")
                
    except RuntimeError as e:
        if "size of tensor" in str(e):
            print(f"Model tensor size mismatch: {e}")
            print("\nThis error suggests your model was exported with different configuration.")
            print("Possible solutions:")
            print("1. Re-export your model with the same configuration used during training")
            print("2. Check if your model expects a different input size")
            print("3. Verify the number of classes matches your training setup")
            print("4. Try using --img-size parameter with different values (e.g., 416, 512, 640)")
            break
        else:
            print(f"Model inference error: {e}")
            continue
    except Exception as e:
        print(f"Unexpected error during inference: {e}")
        continue

    # Post-process predictions
    try:
        detections = postprocess_detections(predictions, original_w, original_h, scale, pad_x, pad_y, min_thresh)
    except Exception as e:
        print(f"Error in postprocessing: {e}")
        detections = []

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

        # Draw label with better formatting
        label = f'{class_name}: {confidence:.2f}'
        labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        label_ymin = max(y1, labelSize[1] + 10)
        cv2.rectangle(display_frame, (x1, label_ymin-labelSize[1]-10), 
                     (x1+labelSize[0], label_ymin+baseLine-10), color, cv2.FILLED)
        cv2.putText(display_frame, label, (x1, label_ymin-7), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    # Calculate and draw framerate
    if source_type in ['video', 'usb', 'picamera']:
        cv2.putText(display_frame, f'FPS: {avg_frame_rate:.1f}', (10, 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    
    # Display detection results with waste category counts
    waste_counts = {}
    for detection in detections:
        class_name = detection['class_name']
        waste_counts[class_name] = waste_counts.get(class_name, 0) + 1
    
    # Display total count
    cv2.putText(display_frame, f'Total waste items: {object_count}', (10, 40), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    
    # Display individual waste type counts
    y_offset = 60
    for waste_type, count in waste_counts.items():
        cv2.putText(display_frame, f'{waste_type}: {count}', (10, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        y_offset += 20
    
    # Display the frame
    cv2.imshow('Waste Detection - YOLOv5 TorchScript', display_frame)
    
    if record: 
        recorder.write(display_frame)

    # Handle key presses
    if source_type in ['image', 'folder']:
        key = cv2.waitKey(0)  # Wait indefinitely for key press
    elif source_type in ['video', 'usb', 'picamera']:
        key = cv2.waitKey(1) & 0xFF  # Non-blocking key check
    
    if key == ord('q') or key == ord('Q'):  # Press 'q' to quit
        break
    elif key == ord('s') or key == ord('S'):  # Press 's' to pause
        print("Paused. Press any key to continue...")
        cv2.waitKey(0)
    elif key == ord('p') or key == ord('P'):  # Press 'p' to save screenshot
        screenshot_name = f'capture_{frame_count:06d}.png'
        cv2.imwrite(screenshot_name, display_frame)
        print(f"Screenshot saved as {screenshot_name}")
    
    # Calculate FPS
    t_stop = time.perf_counter()
    frame_rate_calc = 1.0 / (t_stop - t_start)

    # Update frame rate buffer
    if len(frame_rate_buffer) >= fps_avg_len:
        frame_rate_buffer.pop(0)
    frame_rate_buffer.append(frame_rate_calc)

    # Calculate average FPS
    avg_frame_rate = np.mean(frame_rate_buffer)

# Cleanup
print(f'Processing complete. Average FPS: {avg_frame_rate:.2f}')
if source_type in ['video', 'usb']:
    cap.release()
elif source_type == 'picamera':
    cap.stop()
if record: 
    recorder.release()
cv2.destroyAllWindows()
