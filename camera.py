# -*- coding: utf-8 -*-
"""
Created on Tue Sep 16 13:40:34 2025

@author: fleroux
"""

import datetime
import pathlib
import time
import os
import sys
from abc import ABC, abstractmethod

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from pylablib.devices import DCAM


# %%


class BaseCamera(ABC):

    image_shape: tuple[int, int]
    _max_pixel_value: int

    @abstractmethod
    def get_frame(self):
        pass

    @abstractmethod
    def acquire_mean(self):
        pass

    @abstractmethod
    def close(self):
        pass

    @abstractmethod
    def record_sequence(self, n_frames: int):
        pass


class OrcaCamera(BaseCamera):

    def __init__(
        self,
        exposure_time: float,
        roi: tuple[int, int, int, int],
        binning: int,
        n_frames_buffer: int = 8,
        idx: int = 0,
        serial: str | None = None,
        verbose=True,
    ):

        self._max_pixel_value = 65535

        self.serial = serial
        self.verbose = verbose
        self.exposure_time = exposure_time
        self.roi = roi
        self.binning = binning
        self.n_frames_buffer = n_frames_buffer
        self.verbose = verbose

        if self.serial is not None:
            self.cam = self._find_camera_by_serial(self.serial)
        else:
            self.cam = DCAM.DCAMCamera(idx)
            self.serial = self.cam.get_device_info().serial_number

        self._is_streaming = False

        self.cam.open()

        self._sdk_open_flag = True

        if self.verbose:
            print(f"Connected with ORCA {self.serial}")

        self.start_streaming()

    def start_streaming(self):

        if self._is_streaming:
            return

        self.cam.set_exposure(self.exposure_time)
        self.cam.set_roi(
            hstart=self.roi[0],
            hend=self.roi[1],
            vstart=self.roi[2],
            vend=self.roi[3],
            hbin=self.binning,
            vbin=self.binning,
        )
        self.cam.setup_acquisition("sequence", nframes=self.n_frames_buffer)
        self.cam.start_acquisition()
        time.sleep(0.1)  # let time for pipeline startup
        self.image_shape = tuple(self.cam.get_data_dimensions())
        self._is_streaming = True
        if self.verbose:
            print(f"Acquisition started\n")

    def stop_streaming(self):
        if not self._is_streaming:
            return
        self._is_streaming = False
        self.cam.stop_acquisition()
        if self.verbose:
            print(f"Acquisition stopped")

    def get_frame(self):
        while True:
            frames = self.cam.read_multiple_images()
            if frames:
                if np.any(
                    np.asarray(frames[0]) == self._max_pixel_value
                ):  # check for saturation (max value for uint16)
                    custom_warning("Warning: saturated frame acquired")
                return np.asarray(frames[0], dtype=np.float32)

    def record_sequence(self, n_frames: int):
        timestamps = np.zeros(n_frames, dtype=np.float64)
        frames = []

        count = 0

        while count < n_frames:
            data = self.cam.read_multiple_images(return_info=True)

            if data:
                imgs, infos = data

                for img, info in zip(imgs, infos):
                    frames.append(img)
                    timestamps[count] = info.timestamp_us * 1e-6
                    count += 1

                    if count >= n_frames:
                        break

        frames = np.asarray(frames, dtype=np.float32)
        timestamps -= timestamps[0]  # convert to relative timestamps

        return frames, timestamps

    def acquire_mean(self, n_frames):
        acc = np.zeros(self.image_shape, dtype=np.float32)
        count = 0
        while count < n_frames:
            frames = self.cam.read_multiple_images()
            if frames:
                frames = np.asarray(frames, dtype=np.float32)
                if np.any(
                    frames == self._max_pixel_value
                ):  # check for saturation (max value for uint16)
                    custom_warning("Warning: saturated frame acquired")
                frames = frames[: n_frames - count]
                acc += frames.sum(axis=0)
                count += frames.shape[0]
        return acc / count

    def close(self):
        if self._sdk_open_flag:
            self.stop_streaming()
            self.cam.close()
            self._sdk_open_flag = False
            print(f"End connection with ORCA {self.serial}")
        else:
            print("Camera SDK was already closed")

    def _find_camera_by_serial(self, target_serial: str) -> DCAM.DCAMCamera:
        """
        Search for camera matching the provided serial number.
        """
        n_cams = DCAM.get_cameras_number()

        for idx in range(n_cams):
            cam = DCAM.DCAMCamera(idx=idx)
            info = cam.get_device_info()
            if info.serial_number == target_serial:
                return cam
            cam.close()

        raise RuntimeError(f"Camera with serial '{target_serial}' not found.")

