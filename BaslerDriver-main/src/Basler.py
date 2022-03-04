import os
from pypylon import pylon
import time
import numpy as np
import matplotlib.pyplot as plt
import cv2


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
            grab = self.camera.RetrieveResult(100, pylon.TimeoutHandling_ThrowException)
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
        print(f'{num_images} images acquired in {elapsed:.0f} ms')

        return image_array

    def _convert_BayerBG2RGB(self, image_array: int) -> np.ndarray:
        # Convert a stream of BayerRG8 images to RGB
        num_images = image_array.shape[-1]
        image_out_array = np.zeros((self.img_shape[0], self.img_shape[1], 3, num_images), dtype=int)

        for i in range(num_images):
            image_out_array[0:self.img_shape[0],0:self.img_shape[1], 0:3, i] = \
                cv2.cvtColor(image_array[0:self.img_shape[0],0:self.img_shape[1], i], \
                                cv2.COLOR_BayerBG2RGB)
        return image_out_array

    def get_stream(self, num_images: np.ndarray) -> np.ndarray:
        # Grab a stream of BayerRG8 images and covert them to RGB
        image_array = self._get_raw_stream(num_images)  # Acquire
        image_out_array = self._convert_BayerBG2RGB(image_array)  # Convert from BayerRG8 to RGB
        return image_out_array


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

    


    




