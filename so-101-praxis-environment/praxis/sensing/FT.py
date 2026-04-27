from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Generic, TypeVar, Any, Optional, Dict
import netft_py
from praxis.sensing.base import SensorConfig
from praxis.sensing.base import BaseSensor
@dataclass
class FTConfig(SensorConfig):
    """Base sensor configuration dataclass."""
    name: str = 'ATI-FT-Sensor'
    sample_rate: float = 1000.0  # Hz
    timeout: float = 1.0  # seconds
    ip_address: str = "192.168.1.20"

class FTSensor(BaseSensor[FTConfig]):
    """Force-Torque sensor class."""
    def __init__(self, config: FTConfig):
        """Initialize the sensor."""
        super().__init__(config)
        self.ft_sensor = netft_py.NetFT(config.ip_address)

    def connect(self) -> bool:
        """
        Establish connection to the sensor.
        Returns:
            bool: True if connection successful, False otherwise.
        """
        self.ft_sensor = netft_py.NetFT(self.config.ip_address)
        return True

    def disconnect(self) -> bool:
        """Disconnect from the sensor."""
        self.ft_sensor.close()
        return True

    def stop(self) -> bool:
        """
        Stop the sensor data acquisition.
        Returns:
            bool: True if stopped successfully, False otherwise.
        """
        self.ft_sensor.stop()
        return True

    def read(self) -> Any:
        """
        Read current sensor data.
        Returns:
            Any: The sensor reading (type depends on specific sensor
                implementation).
        """
        return self.ft_sensor.getMeasurement()

    def start(self) -> bool:
        """
        Start the sensor data acquisition.
        Returns:
            bool: True if started successfully, False otherwise.
        """
        return True
      

    @classmethod
    def from_config(cls, config: SensorConfig) -> 'BaseSensor':
        """Create a sensor instance from configuration."""
        return cls(config)