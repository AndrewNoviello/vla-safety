"""Credits: AnySkin, https://github.com/raunaqbhirangi/anyskin, modified for praxis framework by Samuel Jin"""
import time
import numpy as np
import os
from typing import Optional
from anyskin import AnySkinProcess
from praxis.sensing.base import BaseSensor, SensorConfig
from dataclasses import dataclass


def get_port() -> str:
    """Get the port of the AnySkin sensor."""
    for port in os.listdir("/dev"):
        if "ttyACM" in port:
            return f"/dev/{port}"
    raise FileNotFoundError("No AnySkin port found.")


@dataclass
class AnySkinConfig(SensorConfig):
    """AnySkin configuration."""
    name: str = 'anyskin'
    sample_rate: float = 30.0  # Hz
    timeout: float = 1.0  # seconds
    auto_start: bool = True
    
    # AnySkin specific configuration
    port: str = "/dev/ttyACM0"
    num_mags: int = 5


class AnySkinSensor(BaseSensor[AnySkinConfig]):
    """AnySkin sensor implementation."""
    
    def __init__(self, config: AnySkinConfig):
        """Initialize the AnySkin sensor."""
        super().__init__(config)
        self._anyskin_process: Optional[AnySkinProcess] = None

    def connect(self) -> bool:
        """
        Establish connection to the AnySkin sensor.
        
        Returns:
            bool: True if connection successful, False otherwise.
        """
        if self._is_connected:
            return True
        
        try:
            # Initialize AnySkin process
            self._anyskin_process = AnySkinProcess(
                num_mags=self.config.num_mags,
                port=self.config.port,
            )
            
            # Start the background process
            self._anyskin_process.start()
            
            # Wait for sensor to initialize
            time.sleep(1.0)
            
            self._is_connected = True
            
            # Auto-start if configured
            if self.config.auto_start:
                self.start()
            
            return True
            
        except Exception as e:
            self._is_connected = False
            raise RuntimeError(f"Failed to connect to AnySkin sensor: {e}")

    def disconnect(self) -> bool:
        """
        Disconnect from the AnySkin sensor.
        
        Returns:
            bool: True if disconnection successful, False otherwise.
        """
        # Stop if running
        if self._is_running:
            self.stop()
        
        # Stop and cleanup AnySkin process
        if self._anyskin_process is not None:
            
            self._anyskin_process.pause_streaming()
            self._anyskin_process.join()
            self._anyskin_process = None
        
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
        Read current sensor data.
        
        Returns:
            np.ndarray: Magnetometer data of shape (15,) for 5 magnetometers * 3 axes,
                       or None if reading fails.
        """
        if not self._is_connected or self._anyskin_process is None:
            return None
        
        try:
            # get_data returns list of samples, each with [timestamp, ...15 mag values]
            data = self._anyskin_process.get_data(num_samples=1)
            if len(data) == 15:
                #reshape to (5, 3)
                return np.array(data).reshape(5, 3)
            else:
                raise RuntimeError(f"Expected 15 values, got {len(data)}")
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