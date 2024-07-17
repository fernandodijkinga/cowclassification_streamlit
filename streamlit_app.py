import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
import pandas as pd
import collections

device = 'mps'

detector = YOLO("YOLOcowsegmentation_v8m.pt")  # Object detection
classifier_generalhygiene = YOLO("YOLO/mk1_generalhygiene.pt")  # Classify on class 2
classifier_sex = YOLO("YOLO/mk1_sexo.pt")  # Classify on class 2
classifier_age = YOLO("YOLO/mk1_age.pt")  # Classify on class 2
classifier_dairyform = YOLO("YOLO/mk1_dairyform.pt")  # Classify on class 2
classifier_bodydepth = YOLO("YOLO/mk1_bodydepth.pt")  # Classify on class 2
classifier_strength = YOLO("YOLO/mk1_strength.pt")  # Classify on class 2
classifier_stature = YOLO("YOLO/mk1_stature.pt")  # Classify on class 2
classifier_udderattachment = YOLO("YOLO/mk1_udderattachment.pt")  # Classify on class 1
classifier_udderheight = YOLO("YOLO/mk1_udderheight.pt")  # Classify on class 5
classifier_uddercleft = YOLO("YOLO/mk1_uddercleft.pt")  # Classify on class 5
classifier_udderdepth = YOLO("YOLO/mk1_udderdepth.pt")  # Classify on class 5
classifier_udderwidth = YOLO("YOLO/mk1_udderwidth.pt")  # Classify on class 5
classifier_teatlength = YOLO("YOLO/mk1_teatlength.pt")  # Classify on class 5
classifier_rearteat = YOLO("YOLO/mk1_rearteat.pt")  # Classify on class 5
classifier_legside = YOLO("YOLO/mk1_legside.pt")  # Classify on class 4

# Function to run detection and classification and save processed video
def detect_and_classify(video_path, detector, classifiers_frame, classifiers_foreudder, classifiers_rearudder, classifiers_legside):
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    
    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Change codec as needed
    out = cv2.VideoWriter('/tmp/processed.mp4', fourcc, fps, (frame_width, frame_height))
    
    # Initialize counters for predictions
    prediction_counts = collections.defaultdict(collections.Counter)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Run object detection, focusing on classes 1 to 8
        detection_results = detector(frame, conf=0.7, device=device, classes=[1, 2, 3, 4, 5, 6, 7, 8])

        # Iterate over detections
        for result in detection_results:
            if result.boxes:  # Ensure that there are detected boxes
                for box in result.boxes:
                    class_id = int(box.cls)
                    x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                    roi = frame[y1:y2, x1:x2]
                    roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
                    roi_pil = Image.fromarray(roi_rgb)

                    # Classify using the appropriate classifiers based on the detected class
                    text_y = y1  # Initial vertical position for text

                    if class_id == 1:  # Fore Udder
                        for name, classifier in classifiers_foreudder.items():
                            results = classifier(roi_pil, conf=0.8, device=device)
                            if results:
                                for res in results:
                                    if res.probs:
                                        class_name = classifier.names[res.probs.top1]
                                        prediction_counts[name][class_name] += 1
                                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
                                        cv2.putText(frame, f"{name}: {class_name}", (x1, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                                        text_y += 20

                    elif class_id == 2:  # Frame
                        for name, classifier in classifiers_frame.items():
                            results = classifier(roi_pil, conf=0.8, device=device)
                            if results:
                                for res in results:
                                    if res.probs:
                                        class_name = classifier.names[res.probs.top1]
                                        prediction_counts[name][class_name] += 1
                                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
                                        cv2.putText(frame, f"{name}: {class_name}", (x1, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                                        text_y += 20

                    elif class_id == 4:  # Leg Side
                        for name, classifier in classifiers_legside.items():
                            results = classifier(roi_pil, conf=0.8, device=device)
                            if results:
                                for res in results:
                                    if res.probs:
                                        class_name = classifier.names[res.probs.top1]
                                        prediction_counts[name][class_name] += 1
                                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
                                        cv2.putText(frame, f"{name}: {class_name}", (x1, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                                        text_y += 20

                    elif class_id == 5:  # Rear Udder
                        for name, classifier in classifiers_rearudder.items():
                            results = classifier(roi_pil, conf=0.8, device=device)
                            if results:
                                for res in results:
                                    if res.probs:
                                        class_name = classifier.names[res.probs.top1]
                                        prediction_counts[name][class_name] += 1
                                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
                                        cv2.putText(frame, f"{name}: {class_name}", (x1, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                                        text_y += 20

                    else:  # Other detected classes (3, 6, 7, 8)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)

        # Write the frame into the output video
        out.write(frame)

        frame_count += 1
    
    # Release everything if job is finished
    cap.release()
    out.release()
    
    return prediction_counts

# Define classifiers in a dictionary
classifiers_frame = {
    "General Hygiene": classifier_generalhygiene,
    "Sex": classifier_sex,
    "Age": classifier_age,
    "Stature": classifier_stature,
    "Strength": classifier_strength,
    "Body Depth": classifier_bodydepth,
    "Dairy Form": classifier_dairyform
}

classifiers_rearudder = {
    "Udder Height": classifier_udderheight,
    "Udder Width": classifier_udderwidth,
    "Udder Cleft": classifier_uddercleft,
    "Udder Depth": classifier_udderdepth,
    "Rear Teat": classifier_rearteat,
    "Teat Length": classifier_teatlength
    
}

classifiers_foreudder = {
    "Udder Attachment": classifier_udderattachment
}

classifiers_legside = {
    "Leg Side View": classifier_legside
}

# Streamlit app
st.title("Cattle Video Classification")
st.write("Upload a video to process.")

uploaded_video = st.file_uploader("Choose a video...", type=["mp4", "avi", "mov"])

if uploaded_video is not None:
    # Save the uploaded video to a temporary file
    with open("/tmp/uploaded_video.mp4", "wb") as f:
        f.write(uploaded_video.read())

    st.video(uploaded_video)

    if st.button("Process Video"):
        with st.spinner('Processing...'):
            prediction_counts = detect_and_classify("/tmp/uploaded_video.mp4", detector, classifiers_frame, classifiers_foreudder, classifiers_rearudder, classifiers_legside)
        
        st.success('Processing complete!')
        
        # Display the processed video
        st.video("/tmp/processed.mp4")
        
        # Plot the most predicted classes
        st.write("Most Predicted Classes:")
        for classifier_name, counts in prediction_counts.items():
            st.write(f"### {classifier_name}")
            df_counts = pd.DataFrame(list(counts.items()), columns=['Class', 'Count'])
            st.bar_chart(df_counts.set_index('Class'))
