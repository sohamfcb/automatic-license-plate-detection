# License Plate Detection and Text Extraction Web App

This project is a Streamlit web application for detecting license plates in images and extracting the text from the detected license plates using a YOLO model and Tesseract OCR. The application provides an easy-to-use interface for users to upload images, detect license plates, and extract text from them.

## Features

- **Upload Images & Videos**: Users can upload images and videos of cars containing license plates.
- **License Plate Detection**: The app uses a pre-trained YOLO model to detect license plates in the uploaded images.
- **Text Extraction**: Tesseract OCR is used to extract text from the detected license plates.
- **Visualization**: The detected license plates and extracted text are displayed on the web interface.

## Installation

### Prerequisites

- Python 3.7 or higher
- pip (Python package installer)

<!-- #region -->
### Clone the repository:

`git clone https://github.com/sohamfcb/license-plate-detection-app.git`

<!-- #endregion -->
### Set Up Tesseract OCR

#### For Windows:

- Download the Tesseract installer from [here](https://github.com/tesseract-ocr/tesseract/releases).
- Run the installer and add Tesseract to your system PATH during installation.
- Verify the installation by running `tesseract --version` in Command Prompt.

#### For other operating systems, follow the instructions [here](https://github.com/tesseract-ocr/tesseract).


## Usage

### Running the App

`streamlit run app.py`


<!-- #region -->
### Web Interface

- **Upload Image**: Click on the "Browse files" button to upload an image containing a license plate.
- **Detect and Extract**: The app will automatically detect the license plate and extract the text from it.
- **Results**: The detected license plate and extracted text will be displayed on the web interface.

## File Structure

- `app.py`: The main script for running the Streamlit web app.
- `detect.py`: Contains functions for license plate detection and text extraction.
- `requirements.txt`: Lists all the dependencies required for the project.
- `models/`: Directory containing the YOLO model files.
- `images/`: Directory to store sample images.

## Example

[Click here to see an example](https://drive.google.com/file/d/1iqTkO5hDxJDh_nJM_fTLfCkPIi76Z6Lr/view?usp=sharing)


## Acknowledgments

- The YOLO model used in this project is from [YOLOv5](https://github.com/ultralytics/yolov5).
- Tesseract OCR is an open-source OCR engine maintained by [Google](https://github.com/tesseract-ocr).
<!-- #endregion -->
