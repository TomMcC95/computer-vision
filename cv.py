import cv2
import time

#Initialize video capture
cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)

#Initialize video writer
vid_cod = cv2.VideoWriter_fourcc(*'mp4v')
output = cv2.VideoWriter("gopro_stream.MP4", vid_cod, 30, (1080,1920))

#scaling factor (OPTIONAL)
scaling_factor = 0.5

#picture array
picture_array = []
t = time.time()
n = 0

time.sleep(5)

# Loop until you hit the Esc key
while True:
    # Capture the current frame
    ret, frame = cap.read()

# Resize the frame
    #frame = cv2.resize(frame, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_AREA)

# Display and save the image
    cv2.imshow('Webcam', frame)
    output.write(frame)

    if time.time() - t > 6:
        cv2.imwrite(f'{n}.png',frame)
        picture_array.append(f'{n}.png')
    
    n += 1
# Detect if the Esc key has been pressed
    c = cv2.waitKey(1)
    if c == 27:
        break

    if time.time() - t > 7:
        break

# Release the video capture object

#print info to check
height, width, channels = frame.shape
print(height)
print(width)
print(channels)
print(picture_array)
print(frame.dtpye)

cap.release()
output.release()

# Close all active windows
cv2.destroyAllWindows()