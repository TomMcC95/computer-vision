# Package imports
import cv2 as cv # core computer vision library
from pathlib import Path # Provides OS-agnostic way to handle filenames and addresses
import pandas as pd # Pandas handles csv and excel export
import numpy as np

input_directory_name = 'input'
output_directory_name = 'output'

input_image = cv.imread(str(Path.cwd() / input_directory_name / 'Wide.jpg'),1)
cv.namedWindow('window',cv.WINDOW_NORMAL)
cv.resizeWindow('window',960,720)
cv.imshow('window',input_image)
cv.waitKey(0)
cv.destroyAllWindows()