import cv2

#Initialize video capture
cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)

#Initialize video writer
vid_cod = cv2.VideoWriter_fourcc(*'mp4v')
output = cv2.VideoWriter("gopro_stream.MP4", vid_cod, 30, (1080,1920))

#scaling factor (OPTIONAL)
scaling_factor = 0.5

# Loop until you hit the Esc key
while True:
    # Capture the current frame
    ret, frame = cap.read()

# Resize the frame
    frame = cv2.resize(frame, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_AREA)

# Display and save the image
    cv2.imshow('Webcam', frame)
    cv2.imwrite('test.png',frame)
    output.write(frame)
    

# Detect if the Esc key has been pressed
    c = cv2.waitKey(1)
    if c == 27:
        break

# Release the video capture object

height, width, channels = frame.shape
print(height)
print(width)
print(channels)

cap.release()
output.release()

# Close all active windows
cv2.destroyAllWindows()