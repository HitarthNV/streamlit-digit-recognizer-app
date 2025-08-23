# ✍️ Handwritten Digit Recognition App

This project features a web application built with Streamlit that uses a Convolutional Neural Network (CNN) to recognize handwritten digits (0-9). Users can draw a digit on a canvas, and the app will predict the digit along with confidence scores.

## 🚀 Live Demo

Experience the app live on Streamlit Community Cloud:
[https://app-digit-recognizer-app-4qbjdl2jtkxtyyukubdxgm.streamlit.app/](https://app-digit-recognizer-app-4qbjdl2jtkxtyyukubdxgm.streamlit.app/)

## ✨ Features

* **Interactive Drawing Canvas:** Draw digits directly in your web browser.
* **Real-time Prediction:** Get instant predictions as you draw (or on button click).
* **Confidence Scores:** View the probability distribution across all 10 digits.
* **Processed Image Display:** See the 28x28 grayscale image that the CNN receives for prediction, aiding in understanding and debugging.
* **Clear Canvas:** Easily reset the drawing area.

## 🧠 How It Works

The application leverages a pre-trained Convolutional Neural Network (CNN) model. Here's a high-level overview of the process:

1.  **Drawing Input:** Users draw a digit on a `streamlit-drawable-canvas` component, which provides the drawing as an RGBA (Red, Green, Blue, Alpha) image array. The canvas is configured with a black background and white drawing stroke to match the training data's characteristics.
2.  **Image Preprocessing:**
    * The RGBA image is converted to grayscale.
    * It's then resized to $28 \times 28$ pixels, the standard input size for the MNIST dataset.
    * Pixel values are normalized to a 0-1  range.
    * The image is reshaped to match the CNN's expected input format (batch size, height, width, channels).
3.  **Model Prediction:** The preprocessed image is fed into the CNN model, which outputs a probability distribution over the 10 possible digits (0-9).
4.  **Result Display:** The app identifies the digit with the highest probability as the prediction and displays its confidence, along with the probabilities for all other digits.

## 🛠️ Technologies Used

* **Python:** The core programming language.
* **Streamlit:** For building the interactive web application.
* **TensorFlow / Keras:** For building, training, and running the Convolutional Neural Network model.
* **NumPy:** For numerical operations, especially image array manipulation.
* **Pillow (PIL):** For image processing tasks like resizing and format conversion.
* **`streamlit-drawable-canvas`:** A custom Streamlit component for the drawing functionality.

## 🚀 Getting Started (Local Development)

To run this application on your local machine:

### Prerequisites

* Python 3.9+ (recommended)
* `pip` (Python package installer)
* Git

### Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/HitarthNV/streamlit-digit-recognizer-app.git](https://github.com/HitarthNV/streamlit-digit-recognizer-app.git)
 
    ```
    
2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    # On Windows:
    # .\venv\Scripts\activate
    # On macOS/Linux:
    # source venv/bin/activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Ensure the model file is present:**
    Make sure `best_digit_recognition_model.h5` is in the root directory of your project. This file is generated during the model training phase (see Google Colab section above).

### Running the App Locally

Once the dependencies are installed and the model file is in place:

```bash
streamlit run app.py

```

#👋 Connect with Me
* Email: hitarthvasitawrk24@gmail.com

* LinkedIn: [https://www.linkedin.com/in/hitarth-vasita-b39643304/](https://www.linkedin.com/in/hitarth-vasita-b39643304/)
