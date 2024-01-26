import cv2
import torch
from PIL import Image
import torchvision.transforms as T
from transformers import DetrForObjectDetection
import time

# COCO class labels
coco_names = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

# Define the transformation
transform = T.Compose([
    T.Resize(256),  # Resize to 800px on the shortest side
    T.ToTensor(),  # Convert PIL image to PyTorch tensor
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),  # Normalize with ImageNet mean and standard deviation
])

# Load pre-trained model
model = DetrForObjectDetection.from_pretrained('facebook/detr-resnet-50')

# Set the device to GPU if available, CPU for CPU and GPU for GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Generate colors for bounding boxes
colors = {i: tuple(map(int, torch.randint(0, 256, (3,)).tolist())) for i in range(model.config.num_labels - 1)}

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
        

        for box, cl, prob in zip(bboxes.tolist(), clas.tolist(), probas[keep].max(-1).values.tolist()):
            #if cl not in colors:   #This line did not exist
            #    continue           #This line did not exist
            x0, y0, x1, y1 = map(int, box)
            label = coco_names[cl]
            percentage = f"{prob:.2%}"  # Format the probability as a percentage
            label_text = f"{label}: {percentage}"  # Combine label and percentage
            frame = cv2.rectangle(frame, (x0, y0), (x1, y1), colors[cl], 2)
            cv2.putText(frame, label_text, (x0, y0-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, colors[cl], 2)

            # Calculate speed if object exists in the previous frame
            if label in prev_boxes:
                prev_box, prev_time = prev_boxes[label]
                displacement = ((prev_box[0] - x0) ** 2 + (prev_box[1] - y0) ** 2) ** 0.5
                time_diff = time.time() - prev_time
                speed = displacement / time_diff
                speed_text = f"Speed: {speed:.2f} pixels/second"
                cv2.putText(frame, speed_text, (x0, y0-40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, colors[cl], 2)

            # Store current box and timestamp
            curr_boxes[label] = (box, time.time())

        prev_boxes = curr_boxes

        # Display the resulting frame
        cv2.imshow('Object Detection', frame)

        # Quit if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

      # Cleanup
    cap.release()
    cv2.destroyAllWindows()

run_object_detection()




# import cv2
# import torch
# from PIL import Image
# import torchvision.transforms as T
# from transformers import DetrForObjectDetection

# # COCO class labels
# #'__background__',
# coco_names = [
#     '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
#     'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
#     'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
#     'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
#     'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
#     'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
#     'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
#     'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
#     'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
#     'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
#     'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
#     'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
# ]

# # Define the transformation
# transform = T.Compose([
#     T.Resize(300),  # Resize to 800px on the shortest side
#     T.ToTensor(),  # Convert PIL image to PyTorch tensor
#     T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),  # Normalize with ImageNet mean and standard deviation
# ])

# # Load pre-trained model
# model = DetrForObjectDetection.from_pretrained('facebook/detr-resnet-50')

# # Set the device to GPU if available, CPU for CPU and GPU for GPU
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model.to(device)

# # Generate colors for bounding boxes
# colors = {i: tuple(map(int, torch.randint(0, 256, (3,)).tolist())) for i in range(model.config.num_labels - 1)}

# # Preprocess the image
# def preprocess(image):
#     image = Image.fromarray(image).convert('RGB')
#     image = transform(image).unsqueeze(0).to(device)
#     return image

# # Function to convert DETR output format to OpenCV format
# def detr_to_cv2_format(out, frame_size):
#     # Detr model returns bounding boxes in format (center_x, center_y, width, height)
#     center_x, center_y, w, h = out.unbind(1)

#     # Convert box format to (top-left-x, top-left-y, bottom-right-x, bottom-right-y)
#     half_w, half_h = w / 2, h / 2
#     top_left_x, top_left_y = center_x - half_w, center_y - half_h
#     bottom_right_x, bottom_right_y = center_x + half_w, center_y + half_h

#     # Scale bounding boxes from [0, 1] to frame size
#     scale = torch.Tensor([frame_size[1], frame_size[0], frame_size[1], frame_size[0]]).to(device)
#     boxes = torch.stack((top_left_x, top_left_y, bottom_right_x, bottom_right_y), dim=1) * scale

#     return boxes

# # Run object detection
# def run_object_detection():
#     cap = cv2.VideoCapture(0)
#     if not cap.isOpened():
#         print("Unable to access the camera.")
#         return

#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             print("Error capturing frame.")
#             break

#         # Preprocess and perform inference
#         inputs = preprocess(frame)
#         outputs = model(inputs)

#         # Process the predicted results
#         probas = outputs.logits.softmax(-1)[0, :, :-1]
#         keep = probas.max(-1).values > 0.9

#         # Convert bounding boxes from DETR to OpenCV format and draw them on the frame
#         bboxes = detr_to_cv2_format(outputs.pred_boxes[0, keep], frame.shape[:2])
#         clas = probas[keep].max(-1).indices
#         for box, cl, prob in zip(bboxes.tolist(), clas.tolist(), probas[keep].max(-1).values.tolist()):
#             x0, y0, x1, y1 = map(int, box)
#             label = coco_names[cl]
#             percentage = f"{prob:.2%}"  # Format the probability as a percentage
#             label_text = f"{label}: {percentage}"  # Combine label and percentage
#             frame = cv2.rectangle(frame, (x0, y0), (x1, y1), colors[cl], 2)
#             cv2.putText(frame, label_text, (x0, y0-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, colors[cl], 2)

#         # Display the resulting frame
#         cv2.imshow('Object Detection', frame)

#         # Quit if 'q' is pressed
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break

#     # Cleanup
#     cap.release()
#     cv2.destroyAllWindows()

# run_object_detection()