import matplotlib.pyplot as plt
import cv2
import time

#Initialize video capture

cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)

frame_array = []
t = time.time()

#Loop until specific time elapses.
while time.time() - t < 10:

    #Capture the current frame.
    ret, frame = cap.read()

    print(ret)

    #Append the image to frame array.
    frame_array.append(frame)

cap.release()

plt.axis("off")
plt.imshow(cv2.cvtColor(frame_array[-1], cv2.COLOR_BGR2RGB))
plt.show()

