# # app.py
# import streamlit as st
# import numpy as np
# from PIL import Image
# import tensorflow as tf
# from streamlit_drawable_canvas import st_canvas

# # Set a wide layout for better drawing space
# st.set_page_config(layout="centered", page_title="Handwritten Digit Recognition")

# # --- Model Loading ---
# # @st.cache_resource is crucial for efficient model loading in Streamlit
# # It ensures the model is loaded only once across app reruns.
# @st.cache_resource
# def load_model():
#     # Make sure 'best_digit_recognition_model.h5' is in the same directory as app.py
#     try:
#         model = tf.keras.models.load_model('best_digit_recognition_model.h5')
#         return model
#     except Exception as e:
#         st.error(f"Error loading the model: {e}")
#         st.stop() # Stop the app if model can't be loaded

# model = load_model()

# st.title("✍️ Handwritten Digit Recognizer")
# st.markdown("""
# Draw a single digit (0-9) on the canvas below.
# The system will use a Convolutional Neural Network (CNN) to predict what you drew!
# """)

# # --- Drawing Canvas ---
# # Configure background: black, stroke: white to match MNIST-like input
# canvas_result = st_canvas(
#     stroke_width=20,           # Thickness of the drawn line
#     stroke_color="#FFFFFF",    # White color for the drawn digit
#     background_color="#000000", # Black background for the canvas
#     height=280,                # Height of the drawing area
#     width=280,                 # Width of the drawing area
#     drawing_mode="freedraw",   # Allows free-form drawing
#     key="digit_canvas",        # Unique key for the component
# )

# # --- Prediction Button and Logic ---
# if st.button("Predict Digit"):
#     # Check if anything was actually drawn
#     # canvas_result.image_data is an RGBA NumPy array
#     if canvas_result.image_data is not None and np.sum(canvas_result.image_data) > 0:
#         # Get the image data from the canvas
#         img_data = canvas_result.image_data

#         # 1. Convert RGBA to Grayscale
#         # Image.fromarray expects uint8
#         pil_img = Image.fromarray(img_data.astype("uint8")).convert("L")

#         # 2. Resize to 28x28 pixels
#         # Image.LANCZOS is a high-quality downsampling filter
#         pil_img_resized = pil_img.resize((28, 28), Image.LANCZOS)

#         # 3. Convert back to NumPy array
#         img_array = np.array(pil_img_resized)

#         # IMPORTANT: No inversion (255 - img_array) is needed here!
#         # Because we set background_color="#000000" (black) and stroke_color="#FFFFFF" (white)
#         # in `st_canvas`, the image data is already in the desired format:
#         # white digit on a black background, matching standard MNIST models.

#         # 4. Normalize pixel values to 0-1 range
#         img_array = img_array.astype('float32') / 255.0

#         # 5. Reshape for model input: (batch_size, height, width, channels)
#         # The model expects a batch dimension, even for a single image.
#         img_model_input = img_array.reshape(1, 28, 28, 1)

#         # 6. Make prediction
#         prediction = model.predict(img_model_input, verbose=0) # verbose=0 suppresses console output
#         predicted_digit = np.argmax(prediction[0]) # Get the digit with highest probability
#         confidence = np.max(prediction[0]) * 100 # Convert max probability to percentage

#         st.success(f"### Predicted Digit: **{predicted_digit}**")
#         st.info(f"Confidence: **{confidence:.2f}%**")

#         # --- Display Processed Image (for debugging/user feedback) ---
#         st.subheader("Your Drawn Digit (Processed Input to CNN):")
#         # To display, convert the 0-1 normalized array back to 0-255 uint8
#         # and reshape from (1,28,28,1) to (28,28) for PIL conversion.
#         display_img_array = (img_model_input.reshape(28, 28) * 255).astype(np.uint8)
#         display_pil_img = Image.fromarray(display_img_array)
#         st.image(display_pil_img, caption="What the CNN sees (28x28)",  width=100)

#         # --- Display All Probabilities ---
#         st.subheader("All Probabilities:")
#         # Create a list of tuples (digit, probability) and sort them
#         probabilities_list = [(i, prob * 100) for i, prob in enumerate(prediction[0])]
#         probabilities_list.sort(key=lambda x: x[1], reverse=True) # Sort highest first

#         for digit, prob in probabilities_list:
#             st.write(f"Digit {digit}: {prob:.2f}%")

#     else:
#         st.warning("Please draw a digit on the canvas first!")

# st.markdown("---")
import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from streamlit_drawable_canvas import st_canvas
import io  # Required for converting in-memory images to download buttons

# Set a wide layout for better drawing space
st.set_page_config(layout="centered", page_title="Handwritten Digit Recognition")

# --- Model Loading ---
@st.cache_resource
def load_model():
    # Make sure 'best_digit_recognition_model.h5' is in the same directory
    try:
        model = tf.keras.models.load_model('best_digit_recognition_model.h5')
        return model
    except Exception as e:
        st.error(f"Error loading the model: {e}")
        st.stop()

model = load_model()

st.title("✍️ Handwritten Digit Recognizer")
st.markdown("""
Draw a single digit (0-9) on the canvas below.
The system will use a Convolutional Neural Network (CNN) to predict what you drew!
""")

# --- Drawing Canvas ---
canvas_result = st_canvas(
    stroke_width=20,
    stroke_color="#FFFFFF",      # White stroke
    background_color="#000000",  # Black background
    height=280,
    width=280,
    drawing_mode="freedraw",
    key="digit_canvas",
)

# --- NEW: Image Download Button ---
# We check if image_data exists before trying to download
if canvas_result.image_data is not None:
    # 1. Convert the RGBA NumPy array to a PIL Image
    # .astype('uint8') is crucial for PIL to recognize the data type
    img_to_download = Image.fromarray(canvas_result.image_data.astype("uint8"))
    
    # 2. Save the PIL image to an in-memory buffer (virtual file)
    buf = io.BytesIO()
    img_to_download.save(buf, format="PNG")
    byte_im = buf.getvalue()

    # 3. Create the download button
    st.download_button(
        label="📥 Download Drawing",
        data=byte_im,
        file_name="my_digit.png",
        mime="image/png"
    )

# --- Prediction Button and Logic ---
if st.button("Predict Digit"):
    # Check if anything was drawn
    if canvas_result.image_data is not None and np.sum(canvas_result.image_data) > 0:
        
        # Get image data
        img_data = canvas_result.image_data

        # 1. Convert RGBA to Grayscale
        pil_img = Image.fromarray(img_data.astype("uint8")).convert("L")

        # 2. Resize to 28x28 pixels
        pil_img_resized = pil_img.resize((28, 28), Image.LANCZOS)

        # 3. Convert back to NumPy array
        img_array = np.array(pil_img_resized)

        # 4. Normalize pixel values to 0-1 range
        img_array = img_array.astype('float32') / 255.0

        # 5. Reshape for model input: (batch_size, height, width, channels)
        img_model_input = img_array.reshape(1, 28, 28, 1)

        # 6. Make prediction
        prediction = model.predict(img_model_input, verbose=0)
        predicted_digit = np.argmax(prediction[0])
        confidence = np.max(prediction[0]) * 100

        # --- Display Results ---
        st.success(f"### Predicted Digit: **{predicted_digit}**")
        st.info(f"Confidence: **{confidence:.2f}%**")

        st.subheader("What the CNN sees (28x28 input):")
        # Convert back to 0-255 for display purposes
        display_img_array = (img_model_input.reshape(28, 28) * 255).astype(np.uint8)
        display_pil_img = Image.fromarray(display_img_array)
        st.image(display_pil_img, width=100)

        # --- Probabilities Breakdown ---
        st.subheader("Probability Breakdown:")
        probabilities_list = [(i, prob * 100) for i, prob in enumerate(prediction[0])]
        probabilities_list.sort(key=lambda x: x[1], reverse=True)

        for digit, prob in probabilities_list:
            st.write(f"Digit {digit}: {prob:.2f}%")

    else:
        st.warning("Please draw a digit on the canvas first!")

st.markdown("---")
