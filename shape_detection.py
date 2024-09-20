import streamlit as st
import cv2
import numpy as np
from PIL import Image

def shape_detect(image):
    shape_name = ''
    
    # Convert image to grayscale
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply threshold to the grayscale image
    _, thresh = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY)
    
    # Find contours in the thresholded image
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    for contour in contours:
        # Approximate the contour to a polygon
        epsilon = 0.04 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        
        # Number of vertices of the approximated polygon
        vert = len(approx)
        
        # Get bounding rectangle for placing text on the shape
        x, y, w, h = cv2.boundingRect(approx)
        
        # Draw contours
        cv2.drawContours(image, [approx], -1, (0, 255, 0), 3)
        
        # Identify shapes based on the number of vertices
        if vert == 3:
            shape_name = "TRIANGLE"
        elif vert == 4:
            # Calculate aspect ratio for distinguishing square and rectangle
            aspect = w / float(h)
            if 0.95 <= aspect <= 1.05:
                shape_name = "SQUARE"
            else:
                shape_name = "RECTANGLE"
        elif vert == 5:
            shape_name = "PENTAGON"
        elif vert == 6:
            shape_name = "HEXAGON"
        elif vert > 6:
            # Calculate area, perimeter, and circularity to identify circles
            area = cv2.contourArea(contour)
            perimeter = cv2.arcLength(contour, True)
            circularity = 4 * 3.14 * (area / (perimeter ** 2))
            if 0.7 <= circularity <= 1.2:
                shape_name = "CIRCLE"
            else:
                shape_name = "ELLIPSE OR OTHER"
        else:
            shape_name = "UNKNOWN"
        
        # Put the name of the shape on the image
        cv2.putText(image, shape_name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    return image

# Streamlit App
st.title("Shape Detection App")

# Upload image
uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    # Convert the uploaded file to an OpenCV image
    file_bytes = np.asarray(bytearray(uploaded_image.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    
    result_image = shape_detect(image)
    
    # Convert the OpenCV image to PIL format for displaying in Streamlit
    result_image_rgb = cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)
    st.image(result_image_rgb, caption='Processed Image with Shape Detection', use_column_width=True)

    # Provide a download link for the processed image
    result_image_pil = Image.fromarray(result_image_rgb)
    st.download_button(
        label="Download Result Image",
        data=cv2.imencode('.jpg', cv2.cvtColor(np.array(result_image_pil), cv2.COLOR_RGB2BGR))[1].tobytes(),
        file_name="detected_shapes.jpg",
        mime="image/jpeg"
    )
