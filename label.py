import cv2
import numpy as np

# Create a black window
window_size = (1280, 960)
image = np.zeros((window_size[1], window_size[0], 3), dtype=np.uint8)

# Mouse callback function
def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:  # Left click
        print(f'Left click detected at coordinates (x: {x}, y: {y})')
        # Draw a point at click location
        cv2.circle(image, (x, y), 9, (0, 255, 0), -1)

# Add text for quit instructions
font = cv2.FONT_HERSHEY_SIMPLEX
cv2.putText(image, 'q: Quit app', (16, 32), font, 1, (255, 255, 255), 2)
cv2.putText(image, 'q: Quit app', (16, 32), font, 1, (255, 255, 255), 2)

# Create resizable window
cv2.namedWindow('OpenCV Window', cv2.WINDOW_NORMAL)
cv2.setMouseCallback('OpenCV Window', mouse_callback)

while True:
    cv2.imshow('OpenCV Window', image)
    # cv2.resizeWindow('OpenCV Window', window_size[0] // 2, window_size[1] // 2)
    
    # Wait for key press (1ms delay)
    key = cv2.waitKey(1) & 0xFF
    
    # If 'q' is pressed, quit the application
    if key == ord('q'):
        break

# Clean up
cv2.destroyAllWindows()