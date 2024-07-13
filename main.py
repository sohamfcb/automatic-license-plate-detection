import cv2
import pytesseract
import streamlit as st
from ultralytics import YOLO
import os

def process_media(input_path, output_path):
    """
    Process media files based on their type and save the result.

    Parameters:
    - input_path (str): Path to the input media file.
    - output_path (str): Path where the processed media file will be saved.

    Returns:
    - str or None: Output path of the processed media file, or None if processing fails.
    """
    try:
        file_extension = os.path.splitext(input_path)[1].lower()

        if file_extension in ['.mkv', '.mp4']:
            return predict_and_save_video(video_path=input_path, output_video_path=output_path)
        elif file_extension in ['.jpg', '.jpeg', '.png', '.jfif']:
            return predict_and_save_image(image_path=input_path, output_image_path=output_path)
        else:
            st.error(f"Unsupported File Type: {file_extension}")
            return None

    except Exception as e:
        st.error(f"Error processing media: {str(e)}")
        return None

def predict_and_save_image(image_path, output_image_path):
    """
    Predicts license plate numbers from an image and saves annotated image.

    Parameters:
    - image_path (str): Path to the input image file.
    - output_image_path (str): Path where the annotated image will be saved.

    Returns:
    - str or None: Output path of the annotated image, or None if prediction fails.
    """
    try:
        results = model.predict(image_path, device='cpu')
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                confidence = box.conf[0]
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(image, f'{confidence*100:.2f}%', (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

                roi = image[y1:y2, x1:x2]
                text = pytesseract.image_to_string(roi, config='--psm 6')

                if text is not None:
                    st.write(f'Number on License Plate: {text}')

        image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        cv2.imwrite(output_image_path, image)

        return output_image_path

    except Exception as e:
        st.error(f"Error predicting image: {str(e)}")
        return None

def predict_and_save_video(video_path, output_video_path):
    """
    Predicts license plate numbers from a video and saves annotated video.

    Parameters:
    - video_path (str): Path to the input video file.
    - output_video_path (str): Path where the annotated video will be saved.

    Returns:
    - str or None: Output path of the annotated video, or None if prediction fails.
    """
    try:
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            st.error("Video File Not Found")
            return None
        
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = model.predict(rgb_frame, device='cpu')

            for result in results:
                for box in result.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    confidence = box.conf[0]
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f'{confidence*100:.2f}%', (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            out.write(frame)

        cap.release()
        out.release()

        return output_video_path

    except Exception as e:
        st.error(f"Error predicting video: {str(e)}")
        return None

# Initialize Streamlit app
st.title("License Plate Detector")
st.markdown('#### Upload your image or video')
uploaded_file = st.file_uploader("", type=['jpg', 'jpeg', 'png', 'mp4', 'mkv', 'jfif'])

# Initialize YOLO model
model = YOLO('best.pt')
btn=st.button(label='Show')

if uploaded_file is not None:
    input_path = f'temp/{uploaded_file.name}'
    output_path = f'output/{uploaded_file.name}'

    with open(input_path, 'wb') as f:
        f.write(uploaded_file.getbuffer())

    if btn:

        st.write('Processing...')
        result_path = process_media(input_path, output_path)

        if result_path:
            if input_path.endswith('mp4'):
                # video_file=open(result_path,'rb')
                # video_bytes=video_file.read()
                st.download_button(
                    label="Download video",
                    data=open(result_path, 'rb').read(),
                    file_name='output_video.mp4'
                )

            elif input_path.endswith('mkv'):
                st.download_button(
                    label="Download video",
                    data=open(result_path, 'rb').read(),
                    file_name='output_video.mp4'
                )

                # st.video(result_path,format='video/mp4')
            elif input_path.endswith(('jpg','jpeg','png','jfif')):
                st.image(result_path)
                st.download_button(
                    label="Download image",
                    data=open(result_path, 'rb').read(),
                    file_name='output_image.jpg'
                )


            elif input_path.endswith('jpeg'):
                st.image(result_path)
                st.download_button(
                    label="Download image",
                    data=open(result_path, 'rb').read(),
                    file_name='output_image.jpeg'
                )

            elif input_path.endswith('png'):
                st.image(result_path)
                st.download_button(
                    label="Download image",
                    data=open(result_path, 'rb').read(),
                    file_name='output_image.png'
                )

            elif input_path.endswith('jfif'):
                st.image(result_path)
                st.download_button(
                    label="Download image",
                    data=open(result_path, 'rb').read(),
                    file_name='output_image.jfif'
                )