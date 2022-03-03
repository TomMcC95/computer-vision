# Package imports
import time # allows for benchmarking and fps
import cv2 as cv # core computer vision library
from pathlib import Path # Provides OS-agnostic way to handle filenames and addresses
import pandas as pd # Pandas handles csv and excel export
import numpy as np

# Leek Counter- Piotr Geca piotr.geca@npl.co.uk
# --- Instructions ---
# Running the script in live display mode:
#   Pressing 'q' button quits the display and script early (data won't be saved)
#   Pressing 'p' button pauses the script on current frame. Press again to resume.

# File variables
input_directory_name = "input" 
output_directory_name = "output"
results_file_name = 'widths_record'
# video_name = "GH010013.mp4" # the name of directory in the input folder should match the dataset name.
output_path = Path.cwd() / output_directory_name / "GH010012_output.avi"
# video_path = Path.cwd() / input_directory_name / video_name # Pathlib constructing filepath, starting from current working dictionary (where the script file is placed)

video_path = "C:/Users/im7/OneDrive - National Physical Laboratory/A4I/Leeks, Camera, Action/Photos & Videos/2022-02-08 Tom/GH010012.MP4"

# TODO: All files should be checked at start of the script for write permission. If possible, they would be kept open

# Script toggles
toggle_live_display = True # Toggle if the script should run with real-time display function. Slows down execution.
# WARNING: video export broken at the moment
toggle_vidsave = True # Toggle saving live display as video. Warning: toggle_live_display must be True
toggle_progress_bar = True # Activate progress bar function. Requires tqdm package
toggle_background_subtractor = False # Activate a more expensive background subtraction process. Else a simple Otsu threshold is used.
# Note: ROI is now mandatory since most videos show substantial perspective distortion.
## ROI_points = np.array([[549,488],[1253,444],[1304,981],[650,1142]])
# ROI_points = np.array([[587,272],[2427,215],[2481,1947],[641,2005]])
ROI_points = np.array([[540,261],[3486,177],[3560,1949],[526,2010]]) # vlcsnap-2022-02-15-01h14m31s988 and GH010014.mp4
# To obtain good perspective correction, user needs to supply the true dimensions of ROI.
# ROI_true_h_mm = 310 # width of belt
# ROI_true_w_mm = 319
ROI_true_h_mm = 310 # width of belt
ROI_true_w_mm = 507
# Note: With x being the horizontal axis and 0 being on top. 
measure_from_end_px = 100 # Distance, in pixels of the thickness measurement, from the white end.
toggle_output_greyscale = True # Toggle if greyscale signal should be saved alongside auto-thresholded widths
toggle_vertical_travel = False # Toggle if the leeks are travelling vertically. (Horizontal by default)
video_start_fraction = 21/67 # 23/75 # 26/64 # Start and end positions of the video (expressed as fraction of total length)
video_end_fraction = 67/67 # 75/75 # 59/64

# # Original Fine-tuning variables from leek_counter_06.py
# noise_reduction_kernel_size = 10
# surface_smoothing_kernel_size = 10
# signal_channel_no = 2 # Nominate channel to save grey values from. CV uses BGR convention, thus 0=Blue, 1=Green, 2=Red. Note that red channel provides best contrast between white stems and green conveyor.
# signal_width = 150 # padded width of collected greyscale scan values. Must be significantly larger than largest leek.
# minimum_leek_area_px = 8000 # Minimum area (in pixels^2) for the detection to be counted
# maximum_leek_area_px = 25000
# minimum_aspect_ratio = 6 # Miniumum aspect ratio for detection to be measured (rejects overlapping regions and other non-leek-shaped objects)
# maximum_aspect_ratio = 15
# detection_window_positon = 0.5 # 0.5 corresponds to half the frame size. When centerpoint of the detected region passes through, it will be collected
# detection_window_width = 10 # smaller window deals better with crowding but depending on framerate can miss some detections. For leek to be picked up and measured, only one box centerpoint (see demo window output) can be present inside the window.

# Fine-tuning variables
noise_reduction_kernel_size = 10
surface_smoothing_kernel_size = 10
signal_channel_no = 2 # Nominate channel to save grey values from. CV uses BGR convention, thus 0=Blue, 1=Green, 2=Red. Note that red channel provides best contrast between white stems and green conveyor.
signal_width = 600 # padded width of collected greyscale scan values. Must be significantly larger than largest leek.
minimum_leek_area_px = 50000 # Minimum area (in pixels^2) for the detection to be counted
maximum_leek_area_px = 800000
minimum_aspect_ratio = 3 # Miniumum aspect ratio for detection to be measured (rejects overlapping regions and other non-leek-shaped objects)
maximum_aspect_ratio = 30
detection_window_positon = 0.5 # 0.5 corresponds to half the frame size. When centerpoint of the detected region passes through, it will be collected
detection_window_width = 30 # smaller window deals better with crowding but depending on framerate can miss some detections. For leek to be picked up and measured, only one box centerpoint (see demo window output) can be present inside the window.

# --- Code Starts Here ---


# Prepare OpenCV background subtractors
backSub = cv.createBackgroundSubtractorMOG2(history = 750, varThreshold = 20, detectShadows = True)
# backSub = cv.createBackgroundSubtractorKNN(history = 1000, dist2Threshold = 400, detectShadows = True)
# Set capture video
capture = cv.VideoCapture(str(video_path))

# Get dimensions and length of the video
capture.set(cv.CAP_PROP_POS_AVI_RATIO, video_end_fraction)
fheight = int(capture.get(cv.CAP_PROP_FRAME_HEIGHT))
fwidth = int(capture.get(cv.CAP_PROP_FRAME_WIDTH))
fdim = (fheight, fwidth)
frames_total = capture.get(cv.CAP_PROP_POS_FRAMES)
capture.set(cv.CAP_PROP_POS_AVI_RATIO, 0)
# Using CAP_PROP_POS_AVI_RATIO to skip to middle of the video causes error. Use CAP_PROP_POS_FRAMES instead
capture.set(cv.CAP_PROP_POS_FRAMES, frames_total * video_start_fraction)


# Prepare dataframe
dfm = pd.DataFrame(columns=['id', 'width_px', 'width_mm'])  # dataframe with measurements

# Prepare kernels
noise_reduction_kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (noise_reduction_kernel_size,noise_reduction_kernel_size))
surface_smoothing_kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (surface_smoothing_kernel_size,surface_smoothing_kernel_size))
signal_margin = 20
margins_kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (signal_margin,signal_margin))

# Create a frame to store images of detected leeks
# Empty array for temporary storing and transforming of cropped leek
# Temporary leek image storage array is oversized to allow for rotation without risk of cropping
leek_temp = np.zeros((4000,4000,3), dtype = np.uint8)
leek_tight = np.zeros((40,40,3), dtype = np.uint8)
frames_since_detection = 2
# Making array for storing last n leeks:
leek_store = np.zeros((1000,1000,3), dtype = np.uint8) # Doesn't look like this variable is used
leek_result_width = signal_width
leek_result_height = 2500 # 900
leek_result = np.zeros((leek_result_height, leek_result_width,3), dtype = np.uint8)
leek_result_template = leek_result.copy()
leek_result_temp = leek_result.copy()
leek_result_list = [leek_result.copy(), leek_result.copy(),leek_result.copy()]
leek_width = 0
leek_area = 0
leek_width_record = np.empty((0),dtype=np.uint16)
leek_area_record = np.empty((0), dtype=np.uint16)
leek_signal_record = np.empty((0, signal_width), dtype=np.uint8)

# --- Helper Functions ---
# Average red channel values from top and bottom half of the leek image and flip white-side-down. Note opencv convention is BGR
def flip_green_up(img):
    img_masked = np.ma.masked_equal(img, 0)
    red_mean_up = img_masked[img.shape[0]//2:,:,2].mean()
    red_mean_down = img_masked[:img.shape[0]//2,:,2].mean()
    if red_mean_up < red_mean_down: img = cv.rotate(img, cv.ROTATE_180)
    return img
# Remove black margins from image
def crop_to_content(img):
    any_rows = np.any(img[:,:,0], axis=1)
    any_columns = np.any(img[:,:,0], axis=0)
    img_cropped = img[np.ix_(any_rows,any_columns)]
    return img_cropped

def paste_in_center(dst, src):
    y_mid = dst.shape[0] // 2
    x_mid = dst.shape[1] // 2
    h = src.shape[0]
    w = src.shape[1]
    dst[y_mid-int(h/2):y_mid-int(h/2)+h ,x_mid-int(w/2):x_mid-int(w/2)+w] = src
    return dst

def order_points(pts):
	# initialzie a list of coordinates that will be ordered
	# such that the first entry in the list is the top-left,
	# the second entry is the top-right, the third is the
	# bottom-right, and the fourth is the bottom-left
	rect = np.zeros((4, 2), dtype = "float32")
	# the top-left point will have the smallest sum, whereas
	# the bottom-right point will have the largest sum
	s = pts.sum(axis = 1)
	rect[0] = pts[np.argmin(s)]
	rect[2] = pts[np.argmax(s)]
	# now, compute the difference between the points, the
	# top-right point will have the smallest difference,
	# whereas the bottom-left will have the largest difference
	diff = np.diff(pts, axis = 1)
	rect[1] = pts[np.argmin(diff)]
	rect[3] = pts[np.argmax(diff)]
	# return the ordered coordinates
	return rect

def get_transform_params(pts, true_h, true_w):
	# obtain a consistent order of the points and unpack them
	# individually
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    h_w_ratio = true_h / true_w
	# compute the width of the new image, which will be the
	# maximum distance between bottom-right and bottom-left
	# x-coordiates or the top-right and top-left x-coordinates
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
	# Height will be calculated based on the user-supplied ROI dimensions
    maxHeight = int(maxWidth * h_w_ratio)
	# now that we have the dimensions of the new image, construct
	# the set of destination points to obtain a "birds eye view",
	# (i.e. top-down view) of the image, again specifying points
	# in the top-left, top-right, bottom-right, and bottom-left
	# order
    dst = np.array([
		[0, 0],
		[maxWidth - 1, 0],
		[maxWidth - 1, maxHeight - 1],
		[0, maxHeight - 1]], dtype = "float32")
	# compute the perspective transform matrix and then apply it
    M = cv.getPerspectiveTransform(rect, dst)
	# return the warped image
    return (M, (maxWidth, maxHeight))

def get_px_to_mm_factor(transform_params, true_width):
    _, w_h = transform_params
    w_px, _ = w_h
    px_to_mm_factor = true_width / w_px
    return px_to_mm_factor

pts = order_points(ROI_points)
transform_params = get_transform_params(pts, ROI_true_h_mm, ROI_true_w_mm)
px_to_mm_factor = get_px_to_mm_factor(transform_params, ROI_true_w_mm)

_, w_h = transform_params
edge_contact_check = (0,0 ,w_h[0], w_h[1]) # This list will be used later to check if bounding boxes touch edges of the frame
mask = np.ones((w_h[1], w_h[0]), dtype=np.uint8)
leek_mask = np.ones((w_h[1], w_h[0]), dtype=np.uint8)
if toggle_vertical_travel:
    detection_window = [w_h[1]*detection_window_positon - detection_window_width,  w_h[1]*detection_window_positon + detection_window_width]
    detection_window = np.floor(detection_window).astype(np.uint16)
else:
    detection_window = [w_h[0]*detection_window_positon - detection_window_width,  w_h[0]*detection_window_positon + detection_window_width]
    detection_window = np.floor(detection_window).astype(np.uint16)

# Set output (Must be same dimensions as the frames being written to it)
fourcc = cv.VideoWriter_fourcc(*'XVID') # MJPEG, XVID
out = cv.VideoWriter(str(output_path),fourcc, 20.0, (w_h[0], w_h[1]))

# Record time of video processing start
start_time = time.time()

# If progress bars are turned on, set up progress bars
if toggle_progress_bar:
    from tqdm import tqdm
    pbar = tqdm(desc = "Processed frames:", total=frames_total * video_end_fraction - frames_total * video_start_fraction, unit = 'Frames')
    pbar2 = tqdm(desc = "Detected Leeks:", unit = 'Leeks')

framecounter = 0
while True:
    framecounter += 1
    current_frame = capture.get(cv.CAP_PROP_POS_FRAMES)
    ret, frame = capture.read()
    if frame is None:
        break
    if current_frame > (frames_total * video_end_fraction):
        break
    # Apply ROI mask
    frame = cv.warpPerspective(frame, *transform_params)
    # Reduce high-frequency noise with blur
    frame = cv.GaussianBlur(frame, (5,5), 0)
    # Subtract background to get foregroundMask (fgMask)
    if toggle_background_subtractor:
        # Apply fgMask to video
        fgMask = backSub.apply(frame)
        # Keep white only (rejects grey-labeled shadows from background subtractor)
        _, fgMask = cv.threshold(fgMask, 254, 255, 0) # Threshold type 2: Binary Inverted
        # Background subtractor is not perfect at removing object shadow. Use auto-thresholding to create a mask removing dark shadows around detected regions
        frame_masked = cv.bitwise_and(frame, frame, mask = fgMask)
        _, cleanMask = cv.threshold(cv.cvtColor(frame_masked, cv.COLOR_BGR2GRAY), 0, 255, cv.THRESH_BINARY+cv.THRESH_OTSU) # OTSU auto threshold
    else:
        _, cleanMask = cv.threshold(frame[...,2], 0, 255, cv.THRESH_BINARY+cv.THRESH_OTSU) # OTSU auto threshold
    # Morphologival opening to remove small stray detected regions
    cleanMask = cv.morphologyEx(cleanMask, cv.MORPH_OPEN, noise_reduction_kernel)
    # Morphological closing to smooth the leek edge (removes false indents under kernel size)
    cleanMask = cv.morphologyEx(cleanMask, cv.MORPH_CLOSE, surface_smoothing_kernel)
    frame_cleaned = cv.bitwise_and(frame, frame, mask = cleanMask)

    # Use clean binary mask to detect contours of all shapes
    all_contours,hierarchy = cv.findContours(cleanMask, 1, 2)
    contours_indices = []
    boxes = []
    boxes_contours = []
    bounds = []
    # For each detected contour, check the area against minimum, maximum limit.
    for idx, cnt in enumerate(all_contours):
        area = cv.contourArea(cnt)
        if (minimum_leek_area_px <= area <= maximum_leek_area_px): contours_indices.append(idx)
    # For each contour which passed the area check
    for idx in contours_indices:
        cnt = all_contours[idx]
        box = cv.minAreaRect(cnt)
        _, dimensions, _ = box
        # If region passes the aspect ratio test, caclulate bounding box, minimal box and box draw coordinates
        if (maximum_aspect_ratio >= (max(dimensions) / min(dimensions)) >= minimum_aspect_ratio):
            bounding_box = cv.boundingRect(cnt)
            box_contact_points = (bounding_box[0], bounding_box[1], bounding_box[0]+bounding_box[2], bounding_box[1]+bounding_box[3])
            if not any([i == j for i,j in zip(edge_contact_check, box_contact_points)]):
                box_contours = cv.boxPoints(box)
                box_contours = np.int0(box_contours)
                boxes.append(box)
                boxes_contours.append(box_contours)
                bounds.append(bounding_box)
    # If there are boxes detected
    if boxes:
        # For each box in the list
        for idx in range(len(boxes)):
            box = boxes[idx]
            center, dims, angle = box
            # Unless the detected region's center is in the detection window, skip it 
            if toggle_vertical_travel:
                if (center[1] < detection_window[0] or center[1] > detection_window[1]): 
                    continue
            else:
                if (center[0] < detection_window[0] or center[0] > detection_window[1]): 
                    continue
            # Wipe the temporary array clean
            leek_temp[...] = 0
            bounding_box = bounds[idx]
            x,y,w,h = bounding_box
            # Mask frame to only contents of the current small box. This prevents later grabbing neighbouring objects.
            leek_mask[...] = 0
            cv.fillPoly(leek_mask, [boxes_contours[idx]], 255)
            frame_focused = cv.bitwise_and(frame_cleaned, frame_cleaned, mask = leek_mask)
            # This statement rotates leeks to vertical if the smallest-box width was larger than length
            if dims[0] > dims[1]: angle += 90
            max_height, max_width = max(dims), min(dims)
            # Grab contents within bounding box around leek and place it in the oversized leek_temp array for rotation/cropping
            leek_temp[2000-int(h/2):2000-int(h/2)+h ,2000-int(w/2):2000-int(w/2)+w] = frame_focused[y:y+h,x:x+w]
            M = cv.getRotationMatrix2D((2000,2000), angle, 1.0) # Parameters: center, angle, scale
            leek_temp = cv.warpAffine(leek_temp, M, (leek_temp.shape[1], leek_temp.shape[0]))
            # Crop to content
            leek_tight = crop_to_content(leek_temp)
            # Flip image green end up
            leek_tight = flip_green_up(leek_tight)
            # Draw line where measurement is taken
            measurement_row = leek_tight[-measure_from_end_px,:,0]
            edge_left, edge_right = np.argwhere(measurement_row).min(), np.argwhere(measurement_row).max()
            leek_width = edge_right-edge_left
            leek_area = np.count_nonzero(leek_tight[:,:,2])
            # Draw red channel line at the detection row, between edges
            leek_tight[-measure_from_end_px:-measure_from_end_px+2,edge_left:edge_right] = [0,0,255]
            # Signal that in the last frame there was a detection in detection zone
            if toggle_output_greyscale:
                # Dilate the leek mask by margin, to capture the full signal, before auto-threshold
                leek_mask = cv.dilate(leek_mask, kernel = margins_kernel)
                # Increase the size of leek's bounding box by margin
                x,y,w,h = [dim + signal_margin for dim in bounding_box]
                # mask frame to the 
                frame_focused = cv.bitwise_and(frame, frame, mask = leek_mask)
                frame_focused = np.pad(frame_focused, ((200,200),(200,200),(0,0)))
                leek_temp[2000-int(h/2):2000-int(h/2)+h ,2000-int(w/2):2000-int(w/2)+w] = frame_focused[y:y+h,x:x+w]
                leek_temp = cv.warpAffine(leek_temp, M, (leek_temp.shape[1], leek_temp.shape[0]))
                leek_tight2 = crop_to_content(leek_temp)
                leek_tight2 = flip_green_up(leek_tight)
                # Fetch the row at measurement distance, from specified channel
                measurement_row = leek_tight2[-(measure_from_end_px+signal_margin),:,signal_channel_no]
            frames_since_detection = 0
            break # Break statement here will exit the loop after first succesfull detection.
            # Double-detection will be impossible if multible boxes are simultaneously in detection zone. This is a crude fix

    # If this is the last time leek was detected in zone:
    if frames_since_detection == 1:
        leek_result_temp[...] = 0
        # cv.namedWindow('Leek Result Template 1',cv.WINDOW_NORMAL)
        # cv.resizeWindow('Leek Result Template 1',int(leek_result_template.shape[1]/3),int(leek_result_template.shape[0]/3))
        # cv.imshow('Leek Result Template 1',leek_result_template)
        # cv.namedWindow('Leek Tight',cv.WINDOW_NORMAL)
        # cv.resizeWindow('Leek Tight',int(leek_tight.shape[1]/3),int(leek_tight.shape[0]/3))
        # cv.imshow('Leek Tight',leek_tight)
        leek_result_temp = paste_in_center(leek_result_template, leek_tight)
        # cv.namedWindow('Leek Result Template 2',cv.WINDOW_NORMAL)
        # cv.resizeWindow('Leek Result Template 2',int(leek_result_template.shape[1]/3),int(leek_result_template.shape[0]/3))
        # cv.imshow('Leek Result Template 2',leek_result_template)
        leek_width_record = np.append(leek_width_record, leek_width)
        leek_area_record = np.append(leek_area_record, leek_area)
        measurement_row = np.pad(measurement_row, (int((signal_width - measurement_row.size)/2),0))
        measurement_row = np.pad(measurement_row, (0,signal_width - measurement_row.size))
        measurement_row = measurement_row[np.newaxis,:]
        leek_signal_record = np.vstack((leek_signal_record, measurement_row))

    # Compose a picture made of all images in leek_result_list
    frames_since_detection += 1

    # Dimensions of Clean Mask and Demo frames
    frameheight = int(frame.shape[0]/4)
    framewidth = int(frame.shape[1]/4)
    leek_peek_ratio = leek_result_height / leek_result_width

    # Windows to display vids
    cv.namedWindow('Leek Peek',cv.WINDOW_NORMAL)
    cv.resizeWindow('Leek Peek',int(900/leek_peek_ratio),900)
    cv.namedWindow('Clean Mask',cv.WINDOW_NORMAL)
    cv.resizeWindow('Clean Mask',framewidth,frameheight)
    cv.namedWindow('Demo',cv.WINDOW_NORMAL)
    cv.resizeWindow('Demo',framewidth,frameheight)
    
    if toggle_live_display:
        # Decide which frame will be base for the demo
        frame_demo = frame
        # Draw detection window bars
        if toggle_vertical_travel:
            frame_demo[detection_window[0],:,:] = [0,0,255]
            frame_demo[detection_window[1],:,:] = [0,0,255]
        else:
            frame_demo[:,detection_window[0],:] = [0,0,255]
            frame_demo[:,detection_window[1],:] = [0,0,255]
        # Draw smallest boxes and their centers around detected leeks
        for idx in range(len(boxes)):
            box = boxes[idx]
            box_contours = boxes_contours[idx]
            center, _, _ = box
            center = (int(center[0]), int(center[1]))
            cv.drawContours(frame_demo,[box_contours],0,(0,0,255),2)
            cv.circle(frame_demo, center, 5, (0,0,255), 2)
        # Annotate with measurement
            cv.putText(leek_result_temp, "Width(px):" + str(leek_width),
            (0,20), cv.FONT_HERSHEY_DUPLEX, 0.4 , (0,0,255))
            cv.putText(leek_result_temp, "Area(px):" + str(leek_area),
            (0,40), cv.FONT_HERSHEY_DUPLEX, 0.4 , (0,0,255))
        
        # # For diagnosis only!
        # cv.putText(frame_cleaned, f'Detected Leek at angle of {angle} \n Center at: {center}',
        # (15, 15), cv.FONT_HERSHEY_SIMPLEX, 0.5 , (255,255,255))

        # If statement causes the image to only be refreshed on the last frame object is in the detection zone
        cv.imshow('Leek Peek', leek_result_temp)
        cv.imshow('Clean Mask', cleanMask)
        cv.imshow('Demo', frame_demo)
        if toggle_vidsave:
            out.write(frame_demo)
        # Waitkey is necessary only if imshow display is on
        key = cv.waitKey(1)
        if key == ord('q'):
            break
        if key == ord('p'):
            cv.waitKey(-1) #wait until any key is pressed
    if toggle_progress_bar:
        pbar.update(1)
        pbar2.n = leek_width_record.size
        pbar2.update(0)

print(f'Seconds to process the video: {time.time() - start_time}')

df = pd.DataFrame(columns=['width_px', 'area_px'])
df['width_px'] = pd.Series(leek_width_record)
df['area_px'] = pd.Series(leek_area_record)
df['width_mm'] = df['width_px'] * px_to_mm_factor
df.to_csv(results_file_name + ".csv")
np.savetxt(results_file_name + "_signal.csv", leek_signal_record, delimiter=",")


capture.release()
out.release()
cv.destroyAllWindows()