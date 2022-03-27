import cv2
import time
import os
import shutil
import datetime

# Get current working directory and assign to variable.
cwd = os.getcwd()

# Initialize video capture.
cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)

# Create empty frame_array.
frame_array = []

# Sleep while video capture is initializing.
time.sleep(3.5)

# Record start time as a variable.
t = time.time()


# Loop for a specified number of seconds from start time.
while time.time() - t < 15:

    # Capture the current frame
    ret, frame = cap.read()

    # Append the frame
    frame_array.append(frame)
    

# Release the video capture object.
cap.release()

# Close all active windows.
cv2.destroyAllWindows()

# Remove first frame captures as this is just GoPro logo.
frame_array.pop(0)

# Print info to check.
print(f'{len(frame_array)} Frames at {datetime.datetime.today()}')

# Delete folder of frames and recreate.
newpath = f'{cwd}/Frames'
shutil.rmtree(f'{newpath}')
if not os.path.exists(newpath):
    os.makedirs(newpath)

# Save captured frames to check visually.
n=1
for i in frame_array:                 
    cv2.imwrite(f'{cwd}/Frames/{n}.png',i)
    n+=1