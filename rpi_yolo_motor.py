import cv2
import torch
import numpy as np
import RPi.GPIO as GPIO
import time

# --- GPIO Setup ---
IN1, IN2 = 17, 18  # Left motor pins
IN3, IN4 = 22, 23  # Right motor pins

GPIO.setmode(GPIO.BCM)
GPIO.setup([IN1, IN2, IN3, IN4], GPIO.OUT)

def stop():
    GPIO.output(IN1, 0)
    GPIO.output(IN2, 0)
    GPIO.output(IN3, 0)
    GPIO.output(IN4, 0)

def forward():
    GPIO.output(IN1, 1)
    GPIO.output(IN2, 0)
    GPIO.output(IN3, 1)
    GPIO.output(IN4, 0)

def left():
    GPIO.output(IN1, 0)
    GPIO.output(IN2, 1)
    GPIO.output(IN3, 1)
    GPIO.output(IN4, 0)

def right():
    GPIO.output(IN1, 1)
    GPIO.output(IN2, 0)
    GPIO.output(IN3, 0)
    GPIO.output(IN4, 1)

# --- Load YOLOv5 TorchScript model ---
model = torch.jit.load('best.torchscript')
model.eval()

# --- Class names (match your YAML) ---
CLASS_NAMES = ['can', 'bottle', 'box', 'paper', 'plastic', 'glass']

# --- Camera Setup ---
cap = cv2.VideoCapture(0)  # USB Camera

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        img = cv2.resize(frame, (640, 640))
        img_rgb = img[:, :, ::-1]  # BGR to RGB
        img_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).float().div(255).unsqueeze(0)

        with torch.no_grad():
            pred = model(img_tensor)[0]
            pred = pred[pred[:, 4] > 0.4]  # confidence threshold

            # Apply NMS
            boxes = []
            for det in pred:
                cls = int(det[5].item())
                label = CLASS_NAMES[cls]

                if label in ['can', 'plastic']:
                    x1, y1, x2, y2 = det[:4]
                    x_center = ((x1 + x2) / 2).item() / 640

                    # Decide movement
                    if x_center < 0.4:
                        print("Object left – turning left")
                        left()
                    elif x_center > 0.6:
                        print("Object right – turning right")
                        right()
                    else:
                        print("Object center – moving forward")
                        forward()
                    break  # only track first matching object
                else:
                    stop()

        time.sleep(0.1)

except KeyboardInterrupt:
    print("Exiting...")

finally:
    stop()
    cap.release()
    GPIO.cleanup()
