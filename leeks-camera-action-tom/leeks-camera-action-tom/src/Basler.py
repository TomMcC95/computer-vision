import os
from pypylon import pylon
import time
import numpy as np
import matplotlib.pyplot as plt
import cv2
from typing import List


path = os.path.dirname(__file__)
CAMERA_SETTINGS_LOAD = path + "/CameraSettings.txt"
CAMERA_SETTINGS_SAVE = path + "/CameraSettings_saved.txt"


class BaslerCamera():
    """Class to control acquisiton from Basler GigE camera"""

    def __init__(self):

        tl_factory = pylon.TlFactory.GetInstance()
        devices = tl_factory.EnumerateDevices()
        for device in devices:
            nome = device.GetFriendlyName()

        tl_factory = pylon.TlFactory.GetInstance()
        self.camera = pylon.InstantCamera()
        self.camera.Attach(tl_factory.CreateFirstDevice())

        # Load camera settings
        self.camera.Open()
        pylon.FeaturePersistence.Load(CAMERA_SETTINGS_LOAD, self.camera.GetNodeMap())
        # pylon.FeaturePersistence.Save(CAMERA_SETTINGS_SAVE, self.camera.GetNodeMap())
        self.camera.PixelFormat = "BayerRG8" # Camera pixel expressed in BayerRG8
        self.camera.Close()

        # Get shape and type of single raw image
        img = self._get_raw_image()
        self.img_shape = img.shape
        self.img_type = img.dtype

        print(f"Sucessfully connected to camera {nome}")

    def _get_raw_image(self) -> np.ndarray:
        # Grab one single raw image in BayerRG8 format
        self.camera.Open()
        self.camera.StartGrabbing(1)
        grab = self.camera.RetrieveResult(100, pylon.TimeoutHandling_Return)
        if grab.GrabSucceeded():
            img = grab.GetArray()
        self.camera.Close()
        return img

    def get_single_image(self) -> np.ndarray:
        # Grab one single BayerRG8 image and covert it to RGB
        img = self._get_raw_image()
        return cv2.cvtColor(img, cv2.COLOR_BayerBG2RGB) # convert pixels into RGB from BayerRG8

    def _get_raw_stream(self, num_images : int) -> np.ndarray:
        # Grab multiple raw images in BayerRG8 format
        self.camera.Open()
        
        i, failures  = 0, 0
        image_array = np.zeros((*self.img_shape, num_images), dtype = self.img_type)
        print('Acquiring stream of images')

        start = time.time()

        self.camera.StartGrabbing(pylon.GrabStrategy_OneByOne)
        while self.camera.IsGrabbing():
            grab = self.camera.RetrieveResult(1000, pylon.TimeoutHandling_ThrowException)
            if grab.GrabSucceeded():
                # Acquire and convert single image and store it in the image array
                image_array[0:self.img_shape[0],0:self.img_shape[1], i] = grab.GetArray()
                i += 1
            else:
                failures += 1
            if i == num_images:
                break
        
        elapsed = 1000 * (time.time() - start)

        if failures:
            print(f'WARNING: cameara failed to acquire {failures} images')

        self.camera.Close()
        framerate = num_images/elapsed * 1000
        print(f'{num_images} images acquired in {elapsed:.0f} ms. Frame rate of: {framerate:.1f} fps')

        return image_array

    def _convert_BayerBG2RGB(self, image_array: np.ndarray) -> np.ndarray:
        # Convert a stream of BayerRG8 images to RGB
        num_images = image_array.shape[-1]
        image_out_array = np.zeros((self.img_shape[0], self.img_shape[1], 3, num_images), dtype=int)

        for i in range(num_images):
            image_out_array[0:self.img_shape[0],0:self.img_shape[1], 0:3, i] = \
                cv2.cvtColor(image_array[0:self.img_shape[0],0:self.img_shape[1], i], \
                                cv2.COLOR_BayerBG2RGB)[:, :, 2::-1]
        return image_out_array

    def _convert_BayerBG2RGB_toList(self, image_array: np.ndarray) -> List[np.ndarray]:
        # Convert a stream of BayerRG8 images to RGB
        num_images = image_array.shape[-1]
        image_out_list = []

        for i in range(num_images):
            image_out_list.append(
                cv2.cvtColor(image_array[0:self.img_shape[0],0:self.img_shape[1], i], \
                                cv2.COLOR_BayerBG2RGB)[:, :, 2::-1])
        return image_out_list

    def _convert_imgList_to_avi(self, image_list : List[np.ndarray], file : str) -> None:
        # Convert a list of images into a mp4 video
        fps = 10
        size_frame = image_list[0].shape[1::-1]
        print(f'Video image shape: { size_frame}')
        fourcc = 'DIVX'

        # Create video
        video_out = cv2.VideoWriter(file, cv2.VideoWriter_fourcc(*fourcc), fps, size_frame)
        # Add image to video
        for image in image_list:
            video_out.write(image)
        video_out.release()

        return

    def _convert_imgList_to_mp4(self, image_list : List[np.ndarray], file : str) -> None:
        # Convert a list of images into a mp4 video
        fps = 10
        size_frame = image_list[0].shape[1::-1]
        print(f'Video image shape: { size_frame}')
        fourcc = 'mp4v'

        # Create video
        video_out = cv2.VideoWriter(file, cv2.VideoWriter_fourcc(*fourcc), fps, size_frame)
        # Add image to video
        for image in image_list:
            video_out.write(image)
        video_out.release()

        return

    def get_stream_asList(self, num_images: int) -> List[np.ndarray]:
        # Grab a stream of BayerRG8 images and covert them to RGB
        image_array = self._get_raw_stream(num_images)  # Acquire
        image_out_list = self._convert_BayerBG2RGB_toList(image_array)  # Convert from BayerRG8 to RGB
        return image_out_list

    
    def get_stream_asMP4(self, num_images: int, file : str) -> None:
        # Grab a stream of images and covert them to a MP4 video
        image_list = self.get_stream_asList(num_images) # Aquire list of images
        self._convert_imgList_to_mp4(image_list, file) # Convert list of images to mp4
        return
    
    def get_stream_asAVI(self, num_images: int, file : str) -> None:
        # Grab a stream of images and covert them to a MP4 video
        image_list = self.get_stream_asList(num_images) # Aquire list of images
        self._convert_imgList_to_avi(image_list, file) # Convert list of images to avi
        return 


# TODO: currently not working - to fix
    # @property
    # def FrameRate(self) -> float:
    #     return self.camera.AcquisitionFrameRateAbs

    # @FrameRate.setter
    # def FrameRate(self, new_frame_rate: float) -> None:
    #     self.camera.AcquisitionFrameRateAbs = new_frame_rate
    #     return

        
    # @property
    # def ExposureTime(self) -> float:
    #     '''Get camera exposure time'''
    #     return self.camera.ExposureTimeRaw

    # @ExposureTime.setter
    # def ExposureTime(self, new_exposure_time: float) -> None:
    #     '''Set camera exposure time'''
    #     self.camera.ExposureTimeRaw = new_exposure_time
    #     return

    


    




