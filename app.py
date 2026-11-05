import streamlit as st
import cv2
import numpy as np
from streamlit_webrtc import webrtc_streamer
from PIL import Image
import math

st.set_page_config(page_title="Live Roundness Detector", layout="wide")

st.title("🎯 Live Target Roundness Detector")
st.write("Point your camera at the target. Adjust sliders to optimize for your lighting.")

# --- Sidebar Controls ---
st.sidebar.header("Image Processing")
blur_size = st.sidebar.slider("Blur (Smoothness)", 3, 21, 5, step=2)
canny_low = st.sidebar.slider("Canny Low", 0, 255, 50)
canny_high = st.sidebar.slider("Canny High", 0, 255, 150)

st.sidebar.header("Hough Circle Params")
min_radius = st.sidebar.slider("Min Radius", 10, 500, 50)
max_radius = st.sidebar.slider("Max Radius", 501, 1000, 500)
param1 = st.sidebar.slider("Param1 (Edge Sensitivity)", 1, 100, 50)
param2 = st.sidebar.slider("Param2 (Accumulator)", 1, 100, 30)

def calculate_roundness(contour):
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    if perimeter == 0: return 0
    return (4 * math.pi * area) / (perimeter ** 2)

def video_transformer(frame):
    # Convert the frame to BGR (WebRTC sends RGB)
    img = frame.to_ndarray(format="bgr24")
    
    # Pre-processing
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (blur_size, blur_size), 0)
    edged = cv2.Canny(blurred, canny_low, canny_high)

    # Circle Detection
    circles = cv2.HoughCircles(
        edged, 
        cv2.HOUGH_GRADIENT, 
        dp=1, 
        minDist=min_radius, 
        param1=param1, 
        param2=param2, 
        minRadius=min_radius, 
        maxRadius=max_radius
    )

    if circles is not None:
        circles = np.uint16(np.around(circles[0, :])).reshape((-1, 1, 2))
        # Sort by radius descending to find the outer target circle
        circles = circles[circles[:, 2] >= min_radius][:, :2]
        circles = circles[np.argsort(circles[:, 2])[::-1]]

        for i, (x, y) in enumerate(circles):
            radius = int(circles[i, 2])
            # Draw the circle
            cv2.circle(img, (x, y), radius, (0, 255, 0), 3)
            cv2.circle(img, (x, y), 2, (0, 0, 255), 3)

            # Calculate Roundness
            mask = np.zeros(gray.shape, dtype="uint8")
            cv2.circle(mask, (x, y), radius, 255, -1)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                cnt = max(contours, key=cv2.contourArea)
                roundness = calculate_roundness(cnt)
                # Show Roundness on screen
                cv2.putText(img, f"R: {roundness:.2f}", (x-50, y), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

    return img, frame.to_ndarray(format="bgr24")

# --- WebRTC Streamer ---
# This handles the connection between the browser's camera and the Python code
webrtc_streamer(
    key="target_detection",
    video_transformer_factory=lambda: video_transformer,
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

st.info("Instructions: 1. Allow camera access. 2. Adjust sliders until the circles highlight the target correctly.")
