from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Generic, TypeVar, Any, Optional, Dict


@dataclass
class SensorConfig:
    """Base sensor configuration dataclass."""
    name: str = 'base_sensor'
    sample_rate: float = 30.0  # Hz
    timeout: float = 1.0  # seconds


T_SensorConfig = TypeVar('T_SensorConfig', bound=SensorConfig)


class BaseSensor(ABC, Generic[T_SensorConfig]):
    """Base sensor class."""
    def __init__(self, config: T_SensorConfig):
        """Initialize the sensor."""
        self.config = config
        self._is_connected: bool = False
        self._is_running: bool = False

    @abstractmethod
    def connect(self) -> bool:
        """
        Establish connection to the sensor.
        Returns:
            bool: True if connection successful, False otherwise.
        """
        pass

    @abstractmethod
    def disconnect(self) -> bool:
        """
        Disconnect from the sensor.
        Returns:
            bool: True if disconnection successful, False otherwise.
        """
        pass

    @abstractmethod
    def start(self) -> bool:
        """
        Start the sensor data acquisition.
        Returns:
            bool: True if started successfully, False otherwise.
        """
        pass

    @abstractmethod
    def stop(self) -> bool:
        """
        Stop the sensor data acquisition.
        Returns:
            bool: True if stopped successfully, False otherwise.
        """
        pass

    @abstractmethod
    def read(self) -> Any:
        """
        Read current sensor data.
        Returns:
            Any: The sensor reading (type depends on specific sensor
                implementation).
        """
        pass
      
    @property
    def name(self) -> str:
        """Get the name of the sensor."""
        return self.config.name

    @property
    def is_connected(self) -> bool:
        """Check if the sensor is connected."""
        return self._is_connected

    @property
    def is_running(self) -> bool:
        """Check if the sensor is running."""
        return self._is_running


    @classmethod
    def from_config(cls, config: SensorConfig) -> 'BaseSensor':
        """Create a sensor instance from configuration."""
        return cls(config)