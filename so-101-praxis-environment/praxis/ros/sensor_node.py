import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
import numpy as np
from dataclasses import dataclass
from std_msgs.msg import Float64MultiArray, String
from std_srvs.srv import Empty
from praxis.sensing.base import BaseSensor

@dataclass
class SensorNodeConfig:
    """Configuration for SensorNode."""
    update_rate: float = 30.0  # Hz
    sensor_reading_topic: str = 'sensor_reading'
    status_topic: str = 'sensor_status'


class SensorNode(Node):
    """ROS sensor node that binds a BaseSensor to the ROS graph."""

    def __init__(
        self,
        sensor: BaseSensor,
        config: SensorNodeConfig,
    ):
        """
        Initialize the SensorNode.

        Args:
            sensor: The sensor instance to bind to ROS
            config: Configuration for the ROS node
        """
        self.sensor = sensor
        self.config = config
        self.node_name = f'{sensor.name}_node'

        # Initialize ROS node
        super().__init__(self.node_name, namespace=sensor.name)

        # Initialize sensor state
        self._is_connected = False
        self._is_running = False

        # Setup QoS profiles
        self._qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            depth=10
        )

        # Initialize ROS interfaces
        self._setup_publishers()
        self._setup_services()
        self._setup_timers()

        # Start up sensor
        self._startup_sensor()

        self.get_logger().info(
            f'SensorNode {self.node_name} initialized for {sensor.name}'
        )

    def _setup_publishers(self) -> None:
        """Setup ROS publishers."""
        self._sensor_reading_pub = self.create_publisher(
            Float64MultiArray,
            self.config.sensor_reading_topic,
            self._qos_profile
        )

        self._status_pub = self.create_publisher(
            String,
            self.config.status_topic,
            self._qos_profile
        )

    def _setup_services(self) -> None:
        """Setup ROS services."""
        # Connect service
        self._connect_srv = self.create_service(
            Empty,
            'connect_sensor',
            self._connect_service_callback
        )

        # Disconnect service
        self._disconnect_srv = self.create_service(
            Empty,
            'disconnect_sensor',
            self._disconnect_service_callback
        )

        # Start service
        self._start_srv = self.create_service(
            Empty,
            'start_sensor',
            self._start_service_callback
        )

        # Stop service
        self._stop_srv = self.create_service(
            Empty,
            'stop_sensor',
            self._stop_service_callback
        )

    def _setup_timers(self) -> None:
        """Setup ROS timers."""
        # Main update timer
        self._update_timer = self.create_timer(
            1.0 / self.config.update_rate,
            self._update_callback
        )

    def _startup_sensor(self) -> None:
        """Connect to and start the sensor."""
        try:
            self._is_connected = self.sensor.connect()
            if self._is_connected:
                self.get_logger().info(f'Connected to {self.sensor.name}')

                # Start the sensor after connection
                self._is_running = self.sensor.start()
                if self._is_running:
                    self.get_logger().info('Sensor started automatically')
                else:
                    self.get_logger().error('Failed to start sensor')
            else:
                self.get_logger().error(
                    f'Failed to connect to {self.sensor.name}'
                )

        except Exception as e:
            self.get_logger().error(f'Error connecting to sensor: {e}')
            self._is_connected = False

    def _publish_sensor_reading(self) -> None:
        """Publish sensor reading as array."""
        try:
            reading = self.sensor.read()

            # Convert reading to numpy array if not already
            if not isinstance(reading, np.ndarray):
                if isinstance(reading, (list, tuple)):
                    reading = np.array(reading)
                elif isinstance(reading, (int, float)):
                    reading = np.array([reading])
                else:
                    self.get_logger().warn(
                        f'Cannot convert reading of type {type(reading)} '
                        'to array'
                    )
                    return

            msg = Float64MultiArray()
            msg.data = reading.flatten().tolist()

            self._sensor_reading_pub.publish(msg)

        except Exception as e:
            self.get_logger().error(f'Error publishing sensor reading: {e}')

    def _publish_status(self) -> None:
        """Publish sensor status."""
        try:
            status = {
                'connected': self._is_connected,
                'running': self._is_running,
                'name': self.sensor.name,
            }

            msg = String()
            msg.data = str(status)
            self._status_pub.publish(msg)

        except Exception as e:
            self.get_logger().error(f'Error publishing status: {e}')

    def _update_callback(self) -> None:
        """Main update callback to publish sensor readings and status."""
        if not self._is_connected or not self._is_running:
            return

        try:
            self._publish_sensor_reading()
            self._publish_status()

        except Exception as e:
            self.get_logger().error(f'Error in update callback: {e}')

    def _connect_service_callback(self, request, response):
        """Handle connect sensor service."""
        try:
            self._is_connected = self.sensor.connect()
            if self._is_connected:
                self.get_logger().info('Sensor connected via service')
            else:
                self.get_logger().error('Failed to connect to sensor')

        except Exception as e:
            self.get_logger().error(f'Error in connect service: {e}')
            self._is_connected = False

        return response

    def _disconnect_service_callback(self, request, response):
        """Handle disconnect sensor service."""
        try:
            success = self.sensor.disconnect()
            if success:
                self._is_connected = False
                self._is_running = False
                self.get_logger().info('Sensor disconnected via service')
            else:
                self.get_logger().error('Failed to disconnect sensor')

        except Exception as e:
            self.get_logger().error(f'Error in disconnect service: {e}')

        return response

    def _start_service_callback(self, request, response):
        """Handle start sensor service."""
        if not self._is_connected:
            self.get_logger().error('Cannot start sensor: not connected')
            return response

        try:
            self._is_running = self.sensor.start()
            if self._is_running:
                self.get_logger().info('Sensor started via service')
            else:
                self.get_logger().error('Failed to start sensor')

        except Exception as e:
            self.get_logger().error(f'Error in start service: {e}')
            self._is_running = False

        return response

    def _stop_service_callback(self, request, response):
        """Handle stop sensor service."""
        try:
            success = self.sensor.stop()
            if success:
                self._is_running = False
                self.get_logger().info('Sensor stopped via service')
            else:
                self.get_logger().error('Failed to stop sensor')

        except Exception as e:
            self.get_logger().error(f'Error in stop service: {e}')

        return response


def run_sensor_node(sensor: BaseSensor, config: SensorNodeConfig) -> None:
    """Run a SensorNode with ROS2."""
    rclpy.init()

    try:
        node = SensorNode(sensor, config)

        try:
            rclpy.spin(node)
        except KeyboardInterrupt:
            pass
        finally:
            node.destroy_node()

    finally:
        rclpy.shutdown()

