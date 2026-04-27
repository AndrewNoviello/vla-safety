import threading
import time
from collections import namedtuple
import hid


AxisSpec = namedtuple("AxisSpec", ["channel", "byte1", "byte2", "scale"])

SPACE_MOUSE_SPEC = {
    "x": AxisSpec(channel=1, byte1=1, byte2=2, scale=1),
    "y": AxisSpec(channel=1, byte1=3, byte2=4, scale=-1),
    "z": AxisSpec(channel=1, byte1=5, byte2=6, scale=-1),
    "roll": AxisSpec(channel=1, byte1=7, byte2=8, scale=-1),
    "pitch": AxisSpec(channel=1, byte1=9, byte2=10, scale=-1),
    "yaw": AxisSpec(channel=1, byte1=11, byte2=12, scale=1),
}

SpaceMouseState = namedtuple(
    "SpaceMouseState", 
    ["x", "y", "z", "roll", "pitch", "yaw", "left_button", "right_button", "t_last_click"]
)


def to_int16(y1, y2):
    """
    Convert two 8 bit bytes to a signed 16 bit integer.
    Args:
        y1 (int): 8-bit byte
        y2 (int): 8-bit byte
    Returns:
        int: 16-bit integer
    """
    x = (y1) | (y2 << 8)
    if x >= 32768:
        x = -(65536 - x)
    return x


def scale_to_control(x, axis_scale=350.0, min_v=-1.0, max_v=1.0):
    """
    Normalize raw HID readings to target range.
    Args:
        x (int): Raw reading from HID
        axis_scale (float): (Inverted) scaling factor for mapping raw input value
        min_v (float): Minimum limit after scaling
        max_v (float): Maximum limit after scaling
    Returns:
        float: Clipped, scaled input from HID
    """
    x = x / axis_scale
    x = min(max(x, min_v), max_v)
    return x


def convert(b1, b2):
    """
    Converts SpaceMouse message to commands.
    Args:
        b1 (int): 8-bit byte
        b2 (int): 8-bit byte
    Returns:
        float: Scaled value from Spacemouse message
    """
    return scale_to_control(to_int16(b1, b2))


class SpaceMouse:
    """
    A minimalistic driver class for SpaceMouse with HID library.
    Adapted from https://github.com/ARISE-Initiative/robosuite/blob/89a624455d6049a6c17cc524e8643c77aaf8b80e/robosuite/devices/spacemouse.py
    TODO(Yunhai): customize button behavior for e.g. gripper control.
    Args:
        vendor_id (int): HID device vendor id
        product_id (int): HID device product id
    """

    def __init__(self, vendor_id=9583, product_id=50741):

        print("Opening SpaceMouse device...")
        self.device = hid.Device(vendor_id, product_id)
        print(f"SpaceMouse opened (manufacturer: {self.device.manufacturer}, product: {self.device.product}, serial: {self.device.serial})")
        
        self._display_controls()

        # 6-D variables
        self.x, self.y, self.z = 0, 0, 0
        self.roll, self.pitch, self.yaw = 0, 0, 0

        # button states
        self.left_button_state = 0
        self.right_button_state = 0

        # click and hold states
        self.single_click_and_hold = False
        self.t_last_click = 0
        self.elapsed_time = 0

        # launch a new thread to listen to SpaceMouse
        # NOTE(Yunhai): Be careful with ROS + threading interactions. If we see inexplicable space mouse bugs later,
        # this threading might be the culprit.
        self.thread = threading.Thread(target=self.run)
        self.thread.daemon = True
        self.thread.start()

    @staticmethod
    def _display_controls():
        """Method to pretty print controls."""

        def print_command(char, info):
            char += " " * (30 - len(char))
            print("{}\t{}".format(char, info))

        print("")
        print_command("Control", "Command")
        print_command("Right button", "(undefined behavior)")
        print_command("Left button", "(undefined behavior)")
        print_command("Move mouse laterally", "move arm horizontally in x-y plane")
        print_command("Move mouse vertically", "move arm vertically")
        print_command("Twist mouse about an axis", "rotate arm about a corresponding axis")
        print("")

    def _reset_internal_state(self):
        """Resets internal state of controller."""
        self.x, self.y, self.z = 0, 0, 0
        self.roll, self.pitch, self.yaw = 0, 0, 0

        self.single_click_and_hold = False
        self.t_last_click = time.time()

    def start_control(self):
        """
        Method that should be called externally before controller can
        start receiving commands.
        """
        self._reset_internal_state()
        self.lock_state = 0

    def run(self):
        """Method that keeps pulling new messages."""

        while True:
            d = self.device.read(13)

            if d is not None:
                self.parse_readings(d)
    
    def parse_readings(self, d):
        """Parse the readings from the SpaceMouse."""
        # readings from 6-DoF sensor
        if d[0] == 1:
            self.x = -1.0 * convert(d[1], d[2])
            self.y =  1.0 * convert(d[3], d[4])
            self.z = -1.0 * convert(d[5], d[6])

        elif d[0] == 2:
            self.yaw = -1.0 * convert(d[5], d[6])
            self.roll = -1.0 * convert(d[1], d[2])
            self.pitch = 1.0 * convert(d[3], d[4])

        # readings from the side buttons
        elif d[0] == 3:
            # press left button
            if d[1] == 1:
                self.left_button_state = 1
                t_click = time.time()
                self.elapsed_time = t_click - self.t_last_click
                self.t_last_click = t_click
                if self.elapsed_time > 0.5:
                    self.single_click_and_hold = True

            # press right button
            if d[1] == 2 + int(self.single_click_and_hold):
                self.right_button_state = 1
                self.lock_state = 1
            else:
                self.lock_state = 0

            # released button
            if d[1] == 0:
                self.left_button_state = 0
                self.right_button_state = 0
                self.single_click_and_hold = False

    def get_state(self):
        """Get the current state of the SpaceMouse."""
        return SpaceMouseState(
            x=self.x,
            y=self.y,
            z=self.z,
            roll=self.roll,
            pitch=self.pitch,
            yaw=self.yaw,
            left_button=self.left_button_state,
            right_button=self.right_button_state,
            t_last_click=self.t_last_click,
        )


if __name__ == "__main__":
    space_mouse = SpaceMouse()
    space_mouse.start_control()
    while True:
        state = space_mouse.get_state()
        print(state)
        time.sleep(0.01)
