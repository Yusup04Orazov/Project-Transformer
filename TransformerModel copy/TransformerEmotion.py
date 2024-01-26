import cv2
import torch
from PIL import Image
import torchvision.transforms as T
from transformers import DetrForObjectDetection
import time
from deepface import DeepFace


# COCO class labels
coco_names = ['__background__', 'person']

# Define the transformation
transform = T.Compose([
    T.Resize(256),  # Resize to 800px on the shortest side
    T.ToTensor(),  # Convert PIL image to PyTorch tensor
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # Normalize with ImageNet mean and standard deviation
])

# Load pre-trained model
model = DetrForObjectDetection.from_pretrained('facebook/detr-resnet-50')
#facebook trained the model named "detr-resnet-50" for emotion recognition

# Set the device to GPU if available, CPU for CPU and GPU for GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Generate colors for bounding boxes
colors = {i: tuple(map(int, torch.randint(0, 256, (3,)).tolist())) for i in range(model.config.num_labels)}

# Preprocess the image
def preprocess(image):
    image = Image.fromarray(image).convert('RGB')
    image = transform(image).unsqueeze(0).to(device)
    return image

# Function to convert DETR output format to OpenCV format
def detr_to_cv2_format(out, frame_size):
    # Detr model returns bounding boxes in format (center_x, center_y, width, height)
    center_x, center_y, w, h = out.unbind(1)

    # Convert box format to (top-left-x, top-left-y, bottom-right-x, bottom-right-y)
    half_w, half_h = w / 2, h / 2
    top_left_x, top_left_y = center_x - half_w, center_y - half_h
    bottom_right_x, bottom_right_y = center_x + half_w, center_y + half_h

    # Scale bounding boxes from [0, 1] to frame size
    scale = torch.Tensor([frame_size[1], frame_size[0], frame_size[1], frame_size[0]]).to(device)
    boxes = torch.stack((top_left_x, top_left_y, bottom_right_x, bottom_right_y), dim=1) * scale
    return boxes

# Run object detection
def run_object_detection():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Unable to access the camera.")
        return

    prev_boxes = {}  # Store previous frame's bounding boxes and positions

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error capturing frame.")
            break

        # Preprocess and perform inference
        inputs = preprocess(frame)
        outputs = model(inputs)

        # Process the predicted results
        probas = outputs.logits.softmax(-1)[0, :, :-1]
        keep = probas.max(-1).values > 0.9

        # Convert bounding boxes from DETR to OpenCV format and draw them on the frame
        bboxes = detr_to_cv2_format(outputs.pred_boxes[0, keep], frame.shape[:2])
        clas = probas[keep].max(-1).indices

        curr_boxes = {}  # Store current frame's bounding boxes and positions

        for box, prob in zip(bboxes.tolist(), probas[keep].max(-1).values.tolist()):
            x0, y0, x1, y1 = map(int, box)
            cl = 1  # Index for "person" class
            label = coco_names[cl]
            percentage = f"{prob:.2%}"  # Format the probability as a percentage
            label_text = f"{label}: {percentage}"  # Combine label and percentage
            frame = cv2.rectangle(frame, (x0, y0), (x1, y1), colors[cl], 2)
            cv2.putText(frame, label_text, (x0, y0-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, colors[cl], 2)

            try:
                # Detect emotion using DeepFace library
                detected_emotion = DeepFace.analyze(frame, actions=['emotion'])[0]['dominant_emotion']
                emotion_text = f"Emotion: {detected_emotion}"
                cv2.putText(frame, emotion_text, (x0, y0-70), cv2.FONT_HERSHEY_SIMPLEX, 0.9, colors[cl], 2)
            except ValueError as e:
                continue

            # Store current box and timestamp
            curr_boxes[label] = (box, time.time())

        prev_boxes = curr_boxes

        # Display the resulting frame
        cv2.imshow('Object Detection', frame)

        # Break the loop when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the capture and destroy the window
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    run_object_detection()