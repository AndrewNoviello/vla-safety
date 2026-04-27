import os
import re

def get_camera_ids(camera_name: str):
    """
    Find all camera IDs that have the corresponding camera name (Linux version).

    :param camera_name: str; The name of the camera.
    :return: list[int]; List of camera IDs that match the camera name.
    """
    cam_nums = []
    try:
        for file in os.listdir("/sys/class/video4linux"):
            real_file = os.path.realpath("/sys/class/video4linux/" + file + "/name")
            with open(real_file, "rt") as name_file:
                name = name_file.read().rstrip()
            print(name)
            if camera_name in name:
                cam_num = int(re.search(r"\d+$", file).group(0))
                cam_nums.append(cam_num)
                print(f"Camera '{camera_name}' found at /dev/video{cam_num}")
            
        if not cam_nums:
            raise ValueError(f"Camera '{camera_name}' not found. Please check camera name and availability.")
    except Exception as e:
        raise ValueError(f"Error searching for camera: {e}")
    return cam_nums


if __name__ == "__main__":
    camera_name = "DIGIT"
    camera_ids = get_camera_ids(camera_name)
    print(camera_ids)