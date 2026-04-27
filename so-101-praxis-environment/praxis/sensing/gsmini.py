""" Credits: Hung-Jui Joe Huang, https://github.com/joehjhuang/gs_sdk, modified for praxis framework by Samuel Jin"""
import os
import re
import subprocess
from dataclasses import dataclass
from typing import Optional
import numpy as np
import cv2

import ffmpeg


from praxis.sensing.base import BaseSensor, SensorConfig


def get_camera_id(camera_name: str, verbose: bool = True) -> Optional[int]:
    """
    Find the camera ID that has the corresponding camera name (Linux version).

    :param camera_name: str; The name of the camera.
    :param verbose: bool; Whether to print the camera information.
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
                if verbose:
                    found = "FOUND!"
            else:
                if verbose:
                    found = "      "
            if verbose:
                print("{} {} -> {}".format(found, file, name))
    except Exception as e:
        if verbose:
            print(f"Error searching for camera: {e}")
    
    return cam_num


def resize_crop(img: np.ndarray, imgw: int, imgh: int) -> np.ndarray:
    """
    Resize and crop the image to the desired size, removing 1/7 border and cropping from the center.

    :param img: np.ndarray; The image to resize and crop.
    :param imgw: int; The width of the desired image.
    :param imgh: int; The height of the desired image.
    :return: np.ndarray; The resized and cropped image.
    """
    h, w = img.shape[:2]
    
    # Remove 1/7th of border from each side
    border_size_h = int(h * (1 / 7))
    border_size_w = int(np.floor(w * (1 / 7)))
    img = img[border_size_h:h - border_size_h, border_size_w:w - border_size_w]
    
    # Now calculate center crop based on target aspect ratio
    h, w = img.shape[:2]
    target_aspect = imgw / imgh
    current_aspect = w / h
    
    # Calculate crop dimensions to maintain target aspect ratio
    if current_aspect > target_aspect:
        # Image is wider than target, crop width
        crop_w = int(h * target_aspect)
        crop_h = h
        start_x = (w - crop_w) // 2
        start_y = 0
    else:
        # Image is taller than target, crop height
        crop_h = int(w / target_aspect)
        crop_w = w
        start_x = 0
        start_y = (h - crop_h) // 2
    
    # Crop from center
    img = img[start_y:start_y + crop_h, start_x:start_x + crop_w]
    
    # Resize to desired dimensions
    img = cv2.resize(img, (imgw, imgh))
    return img


@dataclass
class GSMiniConfig(SensorConfig):
    """GelSight Mini sensor configuration."""
    name: str = 'gsmini'
    sample_rate: float = 25.0  # Hz
    timeout: float = 1.0  # seconds
    auto_start: bool = True

    id: str = '-1'
    
    # Camera device configuration
    device_name: str = "GelSight Mini"
    imgh: int = 240  # Desired image height
    imgw: int = 320  # Desired image width
    raw_imgh: int = 2464  # Raw camera height
    raw_imgw: int = 3280  # Raw camera width
    framerate: int = 25  # Camera framerate


class GSMiniSensor(BaseSensor[GSMiniConfig]):
    """GelSight Mini tactile sensor implementation with low-latency FFmpeg streaming."""
    
    def __init__(self, config: GSMiniConfig):
        """Initialize the GelSight Mini sensor."""
        super().__init__(config)
        if config.id == '-1':
            self._dev_id = get_camera_id(self.config.device_name)
        else:
            self._dev_id = int(config.id)
        self._device_path: Optional[str] = None
        self._raw_size: int = config.raw_imgh * config.raw_imgw * 3
        self._ffmpeg_process: Optional[subprocess.Popen] = None
        self._ffmpeg_command: Optional[list] = None
        print(f"Camera ID: {self._dev_id}")
        self.connect()
        
    def connect(self) -> bool:
        """
        Establish connection to the GelSight Mini sensor using FFmpeg.
        
        Returns:
            bool: True if connection successful, False otherwise.
        """
        if self._is_connected:
            return True
        
        if self._dev_id is None:
            raise ValueError(
                f"Camera '{self.config.device_name}' not found. "
                "Please check camera name and availability."
            )
        
        self._device_path = f"/dev/video{self._dev_id}"
            
        # Check if ffmpeg binary is available 
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
            
        self._ffmpeg_command = (
            ffmpeg.input(
                self._device_path,
                format="v4l2",
                framerate=self.config.framerate,
                video_size=f"{self.config.raw_imgw}x{self.config.raw_imgh}",
            )
            .output("pipe:", format="rawvideo", pix_fmt="bgr24")
            .global_args("-fflags", "nobuffer")
            .global_args("-flags", "low_delay")
            .global_args("-fflags", "+genpts")
            .global_args("-rtbufsize", "0")
            .compile()
        )
        # Start FFmpeg process
        self._ffmpeg_process = subprocess.Popen(
            self._ffmpeg_command, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE
        )
        
        # Check if process started successfully
        if self._ffmpeg_process.poll() is not None:
            stderr_output = self._ffmpeg_process.stderr.read().decode('utf-8')
            stdout_output = self._ffmpeg_process.stdout.read().decode('utf-8')
            raise RuntimeError(
                f"FFmpeg process failed to start.\nError: {stderr_output}\nOutput: {stdout_output}"
            )
        
        # Warm-up phase: discard the first few frames
        warm_up_frames = 30
        for _ in range(warm_up_frames):
            self._ffmpeg_process.stdout.read(self._raw_size)
        
        self._is_connected = True
        # Auto-start if configured
        if self.config.auto_start:
            self.start()
        
        return True
    
    def disconnect(self) -> bool:
        """
        Disconnect from the GelSight Mini sensor.
        
        Returns:
            bool: True if disconnection successful, False otherwise.
        """
       
        if self._is_running:
            self.stop()
        
        # Release FFmpeg process
        if self._ffmpeg_process is not None:
            try:
                self._ffmpeg_process.stdout.close()
                self._ffmpeg_process.terminate()
                self._ffmpeg_process.wait(timeout=2)
            except Exception as e:
                try:
                    self._ffmpeg_process.kill()
                except:
                    pass
            self._ffmpeg_process = None
        
        self._is_connected = False
        
        return True

    
    def start(self) -> bool:
        """
        Start the sensor data acquisition.
        
        Returns:
            bool: True if started successfully, False otherwise.
        """
        self._is_running = True
        return True

    
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
        Read current sensor data (tactile image) from the FFmpeg stream.
        
        Returns:
            np.ndarray: The tactile image as a numpy array of shape (imgh, imgw, 3) with BGR format,
                       or None if reading fails.
        """
        
        try:
            # Read raw frame data from FFmpeg stdout
            raw_frame = self._ffmpeg_process.stdout.read(self._raw_size)
            
            # Check if we got valid frame data
            if len(raw_frame) == 0:
                raise RuntimeError(
                    "No frame data received from camera. "
                    "Camera may not be accessible or ffmpeg command failed."
                )
            
            if len(raw_frame) != self._raw_size:
                raise RuntimeError(
                    f"Expected {self._raw_size} bytes, got {len(raw_frame)} bytes. "
                    "Raw dimensions may be incorrect."
                )
            
            # Convert raw bytes to numpy array
            frame = np.frombuffer(raw_frame, np.uint8).reshape(
                (self.config.raw_imgh, self.config.raw_imgw, 3)
            )
            
            # Resize and crop to desired dimensions
            frame = resize_crop(frame, self.config.imgw, self.config.imgh)
            
            return frame
            
        except Exception as e:
            return None
    
    
    def __enter__(self):
        """Context manager entry."""
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.disconnect()
        return False
