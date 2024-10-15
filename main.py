import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

# Initialize OpenCV video capture
cap = cv2.VideoCapture(0)

# Check if the camera opened successfully
if not cap.isOpened():
    print("Error: Could not open video stream.")
    exit()

# Set up the Matplotlib figure for 3D visualization
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Initialize plots for head, left hand, and right hand
head_plot, = ax.plot([], [], [], 'bo', label="Head")
hand_left_plot, = ax.plot([], [], [], 'ro', label="Left Hand")
hand_right_plot, = ax.plot([], [], [], 'go', label="Right Hand")

# Set axis limits for 3D plot
ax.set_xlim(0, 640)  # Assuming camera width of 640 pixels
ax.set_ylim(0, 480)  # Assuming camera height of 480 pixels
ax.set_zlim(0, 100)  # Arbitrary Z-axis for visualization
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

# Store motion data (initial positions for head and hands)
motion_data = {
    'head': [320, 240, 50],  # Center of the frame
    'hand_left': [200, 240, 50],  # Approx left-hand position
    'hand_right': [440, 240, 50],  # Approx right-hand position
}

# Function to process the motion detection
def detect_motion(frame, bg_frame):
    # Convert frames to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_bg_frame = cv2.cvtColor(bg_frame, cv2.COLOR_BGR2GRAY)

    # Blur the frames to reduce noise
    gray_frame = cv2.GaussianBlur(gray_frame, (21, 21), 0)
    gray_bg_frame = cv2.GaussianBlur(gray_bg_frame, (21, 21), 0)

    # Compute the absolute difference between the background and the current frame
    diff_frame = cv2.absdiff(gray_bg_frame, gray_frame)

    # Threshold the difference image to get a binary mask
    _, thresh_frame = cv2.threshold(diff_frame, 25, 255, cv2.THRESH_BINARY)

    # Dilate the image to fill in holes
    thresh_frame = cv2.dilate(thresh_frame, None, iterations=2)

    return thresh_frame

# Function to update the 3D plot for each frame
def update_plot(frame_num):
    # Capture the next frame from the camera
    ret, frame = cap.read()
    if not ret:
        return

    # Process motion detection
    motion_mask = detect_motion(frame, bg_frame)

    # Find contours in the motion mask
    contours, _ = cv2.findContours(motion_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # If motion is detected, update the 3D points (head, left hand, right hand)
    if len(contours) > 0:
        for contour in contours:
            if cv2.contourArea(contour) < 1000:
                continue  # Ignore small movements

            # Get the bounding box of the largest motion
            (x, y, w, h) = cv2.boundingRect(contour)

            # Simulate motion in 3D for the head and hands
            motion_data['head'] = [x + w // 2, y + h // 2, np.random.randint(0, 100)]
            motion_data['hand_left'] = [x, y + h // 2, np.random.randint(0, 100)]
            motion_data['hand_right'] = [x + w, y + h // 2, np.random.randint(0, 100)]

    # Update plot data with new motion coordinates
    head_plot.set_data(motion_data['head'][0], motion_data['head'][1])
    head_plot.set_3d_properties(motion_data['head'][2])

    hand_left_plot.set_data(motion_data['hand_left'][0], motion_data['hand_left'][1])
    hand_left_plot.set_3d_properties(motion_data['hand_left'][2])

    hand_right_plot.set_data(motion_data['hand_right'][0], motion_data['hand_right'][1])
    hand_right_plot.set_3d_properties(motion_data['hand_right'][2])

    return head_plot, hand_left_plot, hand_right_plot

# Capture the initial background frame
ret, bg_frame = cap.read()
if not ret:
    print("Error: Could not read background frame from camera.")
    cap.release()
    exit()

# Function to check for the 'q' key press to stop the program
def check_quit():
    if cv2.waitKey(1) & 0xFF == ord('q'):
        plt.close()  # Close the Matplotlib window

# Start the animation
ani = FuncAnimation(fig, update_plot, interval=100)

# Show the Matplotlib plot
plt.legend()
plt.show()

# Release the camera and close windows when quitting
cap.release()
cv2.destroyAllWindows()
