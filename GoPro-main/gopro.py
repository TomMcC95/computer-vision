import matplotlib.pyplot as plt
import cv2
import time

def get_frames(number):

    #Initialize video capture
    cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)

    frame_array = []
    t = time.time()
    
    #Loop until specific time elapses.
    while time.time() - t < (number/10):

        #Capture the current frame.
        ret, frame = cap.read()

        #Append the image to frame array.
        frame_array.append(frame)

    #Return number of frames asked for
    return frame_array[-number:]

frame_array = get_frames(160)

plt.axis("off")
plt.imshow(cv2.cvtColor(frame_array[10], cv2.COLOR_BGR2RGB))
plt.show()

