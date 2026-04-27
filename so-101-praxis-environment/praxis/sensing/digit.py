""" Credits: Hung-Jui Joe Huang, https://github.com/joehjhuang/gs_sdk, modified for praxis framework by Samuel Jin"""
import os
import re
from dataclasses import dataclass
from typing import Optional
import numpy as np
import cv2

from praxis.sensing.base import BaseSensor, SensorConfig

def get_camera_id(camera_name: str) -> Optional[int]:
    """
    Find the camera ID that has the corresponding camera name (Linux version).

    :param camera_name: str; The name of the camera.
    :return: int; The camera ID, or None if not found.
    """
    cam_num = None
    try:
        for file in os.listdir("/sys/class/video4linux"):
            real_file = os.path.realpath("/sys/class/video4linux/" + file + "/name")
            with open(real_file, "rt") as name_file:
                name = name_file.read().rstrip()
            if camera_name in name:
                cam_num = int(re.search(r"\d+$", file).group(0))
            
        if cam_num is None:
            raise ValueError(f"Camera '{camera_name}' not found. Please check camera name and availability.")
        print(f"Camera '{camera_name}' found at /dev/video{cam_num}")
    except Exception as e:
        raise ValueError(f"Error searching for camera: {e}")
    return cam_num

@dataclass
class DigitConfig(SensorConfig):
    """DIGIT sensor configuration."""
    name: str = 'digit'
    sample_rate: float = 15.0  # Hz
    timeout: float = 1.0  # seconds
    auto_start: bool = True
    id: Optional[str] = '-1'

    # Camera device configuration
    device_name: str = "DIGIT"
    imgh: int = 240  # Desired image height
    imgw: int = 320  # Desired image width


class DigitSensor(BaseSensor[DigitConfig]):
    """DIGIT tactile sensor implementation using OpenCV."""
    
    def __init__(self, config: DigitConfig):
        """Initialize the DIGIT sensor."""
        super().__init__(config)
        if config.id == '-1':
            self._dev_id = get_camera_id(self.config.device_name)
        else:
            self._dev_id = int(config.id)
        self._camera: Optional[cv2.VideoCapture] = None
        print(f"Camera ID: {self._dev_id}")
        
    def connect(self) -> bool:
        """
        Establish connection to the DIGIT sensor using OpenCV.
        
        Returns:
            bool: True if connection successful, False otherwise.
        """
        if self._dev_id is None:
            self._dev_id = get_camera_id(self.config.device_name)
            if self._dev_id is None:
                raise ValueError(
                    f"Camera '{self.config.device_name}' not found. "
                    "Please check camera name and availability."
                )
        # Open camera with OpenCV
        self._camera = cv2.VideoCapture(self._dev_id)
        
        if not self._camera.isOpened():
            raise RuntimeError(
                f"Failed to open camera at /dev/video{self._dev_id}. "
                "Camera may be in use or inaccessible."
            )
        # Set FPS to the desired rate
        self._camera.set(cv2.CAP_PROP_FPS, self.config.sample_rate)
        
        for _ in range(10):
            self._camera.read()
        
        self._is_connected = True
        # Auto-start
        self.start()
        
        return True
    
    def disconnect(self) -> bool:
        """
        Disconnect from the DIGIT sensor.
        
        Returns:
            bool: True if disconnection successful, False otherwise.
        """
        # Stop if running
        if self._is_running:
            self.stop()
        # Release camera
        if self._camera is not None:
            self._camera.release()
            self._camera = None
        
        self._is_connected = False
        return True

    
    def start(self) -> bool:
        """
        Start the sensor data acquisition.
        
        Returns:
            bool: True if started successfully, False otherwise.
        """
            
        if self._is_running:
            return True
        try:
            self._is_running = True
            return True
        except Exception as e:
            self._is_running = False
            return False

    
    def stop(self) -> bool:
        """
        Stop the sensor data acquisition.
        
        Returns:
            bool: True if stopped successfully, False otherwise.
        """
        
        try:
            self._is_running = False
            return True
        except Exception as e:
            return False
    
    def read(self) -> Optional[np.ndarray]:
        """
        Read current sensor data (tactile image) from the camera.
        
        Returns:
            np.ndarray: The tactile image as a numpy array of shape (imgh, imgw, 3) with BGR format,
                       or None if reading fails.
        """
        
        try:
            # Read raw frame data from camera
            ret, raw_frame = self._camera.read()
            if not ret:
                raise RuntimeError("Failed to read frame from camera.")
            raw_frame = cv2.resize(raw_frame, (self.config.imgw, self.config.imgh))
            return raw_frame
            
        except Exception as e:
            raise RuntimeError(f"Failed to read frame from camera: {e}")
    
    def __enter__(self):
        """Context manager entry."""
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.disconnect()
        return False

