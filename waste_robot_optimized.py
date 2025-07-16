import torch
import numpy as np
import RPi.GPIO as GPIO
import time
from picamera2 import Picamera2
import cv2
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class WasteCollectionRobot:
    def __init__(self):
        # GPIO Setup
        self.IN1, self.IN2 = 17, 18  # Left motor pins
        self.IN3, self.IN4 = 22, 23  # Right motor pins
        
        GPIO.setmode(GPIO.BCM)
        GPIO.setup([self.IN1, self.IN2, self.IN3, self.IN4], GPIO.OUT)
        
        # Class names (match your training YAML)
        self.CLASS_NAMES = ['can', 'plastic bottle', 'salvinia', 'water lettuce', 'plastic bag', 'water hyacinth']
        self.TARGET_CLASSES = {'can', 'plastic bottle', 'salvinia', 'water lettuce', 'plastic bag', 'water hyacinth'}
        
        # Detection parameters
        self.CONFIDENCE_THRESHOLD = 0.4
        self.LEFT_THRESHOLD = 0.4
        self.RIGHT_THRESHOLD = 0.6
        self.IMAGE_WIDTH = 640
        
        # Initialize camera and model
        self.setup_camera()
        self.load_model()
        
    def setup_camera(self):
        """Initialize Picamera2 with error handling"""
        try:
            self.picam2 = Picamera2()
            config = self.picam2.create_preview_configuration(
                main={"format": "RGB888", "size": (self.IMAGE_WIDTH, self.IMAGE_WIDTH)}
            )
            self.picam2.configure(config)
            self.picam2.start()
            logger.info("Camera initialized successfully")
            
            # Allow camera to warm up
            time.sleep(2)
        except Exception as e:
            logger.error(f"Camera initialization failed: {e}")
            raise
    
    def load_model(self):
        """Load YOLOv5 model with error handling"""
        try:
            self.model = torch.jit.load('best.torchscript')
            self.model.eval()
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Model loading failed: {e}")
            raise
    
    def stop(self):
        """Stop all motors"""
        GPIO.output([self.IN1, self.IN2, self.IN3, self.IN4], GPIO.LOW)
    
    def forward(self):
        """Move forward"""
        GPIO.output(self.IN1, GPIO.HIGH)
        GPIO.output(self.IN2, GPIO.LOW)
        GPIO.output(self.IN3, GPIO.HIGH)
        GPIO.output(self.IN4, GPIO.LOW)
    
    def turn_left(self):
        """Turn left"""
        GPIO.output(self.IN1, GPIO.LOW)
        GPIO.output(self.IN2, GPIO.HIGH)
        GPIO.output(self.IN3, GPIO.HIGH)
        GPIO.output(self.IN4, GPIO.LOW)
    
    def turn_right(self):
        """Turn right"""
        GPIO.output(self.IN1, GPIO.HIGH)
        GPIO.output(self.IN2, GPIO.LOW)
        GPIO.output(self.IN3, GPIO.LOW)
        GPIO.output(self.IN4, GPIO.HIGH)
    
    def preprocess_image(self, img):
        """Convert image to tensor format for inference"""
        img_tensor = torch.from_numpy(img).permute(2, 0, 1).float().div(255).unsqueeze(0)
        return img_tensor
    
    def detect_objects(self, img_tensor):
        """Run inference and return filtered detections"""
        with torch.no_grad():
            predictions = self.model(img_tensor)[0]
            # Filter by confidence threshold
            confident_preds = predictions[predictions[:, 4] > self.CONFIDENCE_THRESHOLD]
            return confident_preds
    
    def process_detections(self, detections):
        """Process detections and return movement command"""
        for detection in detections:
            cls_id = int(detection[5].item())
            if cls_id < len(self.CLASS_NAMES):
                label = self.CLASS_NAMES[cls_id]
                
                if label in self.TARGET_CLASSES:
                    # Calculate object center position
                    x1, y1, x2, y2 = detection[:4]
                    x_center = ((x1 + x2) / 2).item() / self.IMAGE_WIDTH
                    confidence = detection[4].item()
                    
                    logger.info(f"Detected {label} at x_center: {x_center:.2f}, confidence: {confidence:.2f}")
                    
                    # Return movement command based on position
                    if x_center < self.LEFT_THRESHOLD:
                        return "left"
                    elif x_center > self.RIGHT_THRESHOLD:
                        return "right"
                    else:
                        return "forward"
        
        return "stop"
    
    def execute_movement(self, command):
        """Execute movement based on command"""
        if command == "left":
            logger.info("Object left - turning left")
            self.turn_left()
        elif command == "right":
            logger.info("Object right - turning right")
            self.turn_right()
        elif command == "forward":
            logger.info("Object center - moving forward")
            self.forward()
        else:
            logger.info("No matching object detected - stopping")
            self.stop()
    
    def run(self, display=False):
        """Main execution loop"""
        logger.info("Starting waste collection robot...")
        
        try:
            while True:
                # Capture frame
                img = self.picam2.capture_array()
                
                # Preprocess for inference
                img_tensor = self.preprocess_image(img)
                
                # Detect objects
                detections = self.detect_objects(img_tensor)
                
                # Process detections and get movement command
                movement_command = self.process_detections(detections)
                
                # Execute movement
                self.execute_movement(movement_command)
                
                # Optional display (only enable for debugging with monitor)
                if display:
                    cv2.imshow("Robot Vision", img)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                
                # Small delay to prevent excessive CPU usage
                time.sleep(0.1)
                
        except KeyboardInterrupt:
            logger.info("Interrupted by user. Exiting...")
        except Exception as e:
            logger.error(f"Runtime error: {e}")
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Clean up resources"""
        logger.info("Cleaning up...")
        self.stop()
        self.picam2.stop()
        if hasattr(cv2, 'destroyAllWindows'):
            cv2.destroyAllWindows()
        GPIO.cleanup()

# Usage
if __name__ == "__main__":
    robot = WasteCollectionRobot()
    
    # Run without display to avoid V4L2 warnings (for headless operation)
    robot.run(display=False)
    
    # Enable display only when connected to monitor for debugging
    # robot.run(display=True)
