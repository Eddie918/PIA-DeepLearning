#YOLO web app created using streamlit and using a pre-trained model from huggingface
from transformers import YolosImageProcessor, YolosForObjectDetection
from PIL import Image, ImageDraw
import torch
import time
import streamlit as st
import cv2




# Load the model

model_name = 'hustvl/yolos-tiny'
model = YolosForObjectDetection.from_pretrained(model_name)
image_processor = YolosImageProcessor.from_pretrained(model_name)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


name = st.text_input("Hi, welcome to the YOLO Object Detection App. What's your name? ")


# Title 
st.title("Object recognition using YOLO model")
if name:
    st.subheader(f"Welcome {name}, feel free to upload your image to see how YOLO works! :)")

# Sidebar   
st.sidebar.title("About")
st.sidebar.info("YOLO (You Only Look Once) is a real-time object detection system. This app uses a pre-trained YOLO model to detect objects in images.")


    

# Upload image
uploaded_file = st.file_uploader("Upload the desired image", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    # Display uploaded image
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Process the image
    inputs = image_processor(images=image, return_tensors="pt").to(device)
    outputs = model(**inputs)

    # Post-process detections
    target_sizes = torch.tensor([image.size[::-1]])
    results = image_processor.post_process_object_detection(outputs, threshold=0.3, target_sizes=target_sizes)[0]

    # Draw bounding boxes
    draw = ImageDraw.Draw(image)
    detected_objects = []
    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        box = [round(i, 2) for i in box.tolist()]
        label_str = model.config.id2label[label.item()]
        detected_objects.append(f"{label_str} ({round(score.item(), 2)})")
        draw.rectangle(box, outline="red", width=3)
        draw.text((box[0], box[1] - 10), f"{label_str}: {round(score.item(), 2)}", fill="red")

    # Display the processed image with detections
    st.image(image, caption="Processed Image with Detections", use_column_width=True)

    # Display detected objects below the processed image
    st.markdown("### Detected Objects")
    if detected_objects:
        for obj in detected_objects:
            st.write(f"- {obj}")
    else:
        st.write("No objects detected.")

# Upload video
uploaded_video = st.file_uploader("Upload the desired video", type=["mp4", "avi", "mov"])
if uploaded_video is not None:
    temp_video_path = "temp_video.mp4"
    with open(temp_video_path, "wb") as f:
        f.write(uploaded_video.read())

    # Display the uploaded video
    st.video(temp_video_path)

    # Process video
    cap = cv2.VideoCapture(temp_video_path)
    frame_placeholder = st.empty()
    frame_skip = 0  # Dynamic frame skipping

    while cap.isOpened():
        start_time = time.time()

        ret, frame = cap.read()
        if not ret:
            break

        # Skip frames dynamically
        if frame_skip > 0:
            frame_skip -= 1
            continue

        # Resize frame
        frame_resized = cv2.resize(frame, (640, 360))
        frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(frame_rgb)

        # YOLO inference
        inputs = image_processor(images=image, return_tensors="pt").to(device)
        outputs = model(**inputs)

        # Post-process detections
        target_sizes = torch.tensor([image.size[::-1]])
        results = image_processor.post_process_object_detection(outputs, threshold=0.3, target_sizes=target_sizes)[0]

        # Draw bounding boxes
        draw = ImageDraw.Draw(image)
        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            box = [round(i, 2) for i in box.tolist()]
            label_str = model.config.id2label[label.item()]
            draw.rectangle(box, outline="red", width=3)
            draw.text((box[0], box[1] - 10), f"{label_str}: {round(score.item(), 2)}", fill="red")

        # Display the frame
        frame_placeholder.image(image, use_column_width=True)

        # Calculate processing time and adjust frame skipping
        processing_time = time.time() - start_time
        frame_skip = int(processing_time / (1 / 30))  # Targeting 30 FPS

    cap.release()
    st.success("Video processing completed!")