import numpy as np
import threading
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
from praxis import MODELS_DIR
import os
import mujoco
from mujoco import MjData, MjSpec

SO101_XML_PATH = os.path.join(MODELS_DIR, "so101/mjcf/so101_new_calib.xml")
SO101_EE_SITE_NAME = "gripperframe"
SO101_DOF = 6

TABLE_Z_MIN = 0.076
INTERPOLATION_STEPS = 80
WAYPOINT_HOLD_STEPS = 20
SELF_COLLISION_MARGIN = 0.001

WORKSPACE_MIN = np.array([0.05, -0.20, TABLE_Z_MIN])
WORKSPACE_MAX = np.array([0.25,  0.20,  0.30])

IK_RANDOM_SAMPLES = 300
IK_MAX_ATTEMPTS   = 10

# Gripper randomisation
GRIPPER_CHANGE_TICKS = 60
GRIPPER_INTERP_TICKS = 40
GRIPPER_MIN_PCT = 0.0
GRIPPER_MAX_PCT = 100.0


class RandomEEMover(Node):
    def __init__(self):
        super().__init__(node_name="random_ee_mover")
        self.get_logger().set_level(rclpy.logging.LoggingSeverity.FATAL)

        self.pub_joints = self.create_publisher(
            msg_type=Float64MultiArray,
            topic="so101real/joint_command",
            qos_profile=10,
        )

        self._init_mujoco()
        self._q_current = self._neutral_percentage()
        self.get_logger().info(f"Neutral: {np.round(self._q_current, 1).tolist()}")

        self._q_target: np.ndarray | None = None
        self._interp_step = 0
        self._hold_step   = 0
        self._phase       = "new_target"

        self._next_q: np.ndarray | None = None
        self._generating = False
        self._gen_lock   = threading.Lock()

        # CONFIG: Based on your diagnostic data, the camera is the X-axis (Index 0)
        self.CAMERA_AXIS_IDX = 0 

        self._gripper_current = 50.0
        self._gripper_target  = 50.0
        self._gripper_step    = 0
        self._gripper_change_countdown = GRIPPER_CHANGE_TICKS

        self.timer = self.create_timer(0.025, self._timer_callback)
        self.get_logger().info("RandomEEMover ready")

    def _init_mujoco(self):
        if not os.path.exists(SO101_XML_PATH):
            raise FileNotFoundError(f"MuJoCo XML not found: {SO101_XML_PATH}")
        self._spec     = MjSpec.from_file(SO101_XML_PATH)
        self._model    = self._spec.compile()
        self._data     = MjData(self._model)
        self._data_gen = MjData(self._model)
        self._ee_site_id = mujoco.mj_name2id(
            self._model, mujoco.mjtObj.mjOBJ_SITE, SO101_EE_SITE_NAME
        )
        self._lower_rad = self._model.jnt_range[:SO101_DOF, 0].copy()
        self._upper_rad = self._model.jnt_range[:SO101_DOF, 1].copy()

    def _neutral_percentage(self) -> np.ndarray:
        q = np.zeros(SO101_DOF)
        q[SO101_DOF - 1] = 50.0
        return q

    def _percentage_to_radians(self, q_pct: np.ndarray) -> np.ndarray:
        q_rad = np.zeros(SO101_DOF)
        for i in range(SO101_DOF - 1):
            q_rad[i] = (self._lower_rad[i] + ((q_pct[i] + 100) / 200.0) * (self._upper_rad[i] - self._lower_rad[i]))
        g = SO101_DOF - 1
        q_rad[g] = (self._lower_rad[g] + (q_pct[g] / 100.0) * (self._upper_rad[g] - self._lower_rad[g]))
        return q_rad

    def _radians_to_percentage(self, q_rad: np.ndarray) -> np.ndarray:
        q_pct = np.zeros(SO101_DOF)
        for i in range(SO101_DOF - 1):
            span    = self._upper_rad[i] - self._lower_rad[i]
            q_pct[i] = ((q_rad[i] - self._lower_rad[i]) / span) * 200.0 - 100.0
        g    = SO101_DOF - 1
        span = self._upper_rad[g] - self._lower_rad[g]
        q_pct[g] = ((q_rad[g] - self._lower_rad[g]) / span) * 100.0
        return q_pct

    def _has_self_collision(self, mjdata: MjData) -> bool:
        for i in range(mjdata.ncon):
            if mjdata.contact[i].dist < -SELF_COLLISION_MARGIN:
                return True
        return False

    def _ee_above_table(self, mjdata: MjData) -> bool:
        return float(mjdata.site(self._ee_site_id).xpos[2]) >= TABLE_Z_MIN

    def _start_generating(self):
        with self._gen_lock:
            if self._generating: return
            self._generating = True
        threading.Thread(target=self._gen_thread, daemon=True).start()

    def _gen_thread(self):
        result = None
        for _ in range(IK_MAX_ATTEMPTS):
            target_pos = np.random.uniform(WORKSPACE_MIN, WORKSPACE_MAX)
            best_q_rad = None
            best_dist  = np.inf

            for _ in range(IK_RANDOM_SAMPLES):
                q_rad = np.random.uniform(self._lower_rad, self._upper_rad)
                self._data_gen.qpos[:SO101_DOF] = q_rad
                mujoco.mj_forward(self._model, self._data_gen)

                if self._has_self_collision(self._data_gen): continue
                if not self._ee_above_table(self._data_gen): continue

                # --- ORIENTATION CONSTRAINTS (tighter) ---
                xmat = self._data_gen.site(self._ee_site_id).xmat.reshape(3, 3)
                view_vec = xmat[:, self.CAMERA_AXIS_IDX]
                vx, vy, vz = float(view_vec[0]), float(view_vec[1]), float(view_vec[2])

                # 1) Side-to-side: tighten (smaller = less left/right)
                MAX_SIDE = 0.25
                if abs(vy) > MAX_SIDE:
                    continue

                # 2) Prefer downward: require at least a bit of downward tilt.
                #    vz < 0 means pointing downward; the more negative, the more downward.
                MIN_DOWN = -0.15
                if vz > MIN_DOWN:
                    continue

                # 3) If you still want to prevent “backward” when near-horizontal,
                #    keep this (optional with the MIN_DOWN rule, but harmless):
                if vx < 0.0:
                    continue
                # --- END ORIENTATION CONSTRAINTS ---

                dist = np.linalg.norm(self._data_gen.site(self._ee_site_id).xpos - target_pos)
                if dist < best_dist:
                    best_dist  = dist
                    best_q_rad = q_rad.copy()

            if best_q_rad is not None and best_dist < 0.10:
                result = self._radians_to_percentage(best_q_rad)
                break

        with self._gen_lock:
            self._next_q = result
            self._generating = False

        if result is not None:
            pass

    def _update_gripper(self) -> float:
        self._gripper_change_countdown -= 1
        if self._gripper_change_countdown <= 0:
            self._gripper_target = float(np.random.uniform(GRIPPER_MIN_PCT, GRIPPER_MAX_PCT))
            self._gripper_step = 0
            self._gripper_change_countdown = GRIPPER_CHANGE_TICKS

        if self._gripper_step < GRIPPER_INTERP_TICKS:
            t = self._gripper_step / GRIPPER_INTERP_TICKS
            t_smooth = t * t * (3.0 - 2.0 * t)
            self._gripper_current = (self._gripper_current + t_smooth * (self._gripper_target - self._gripper_current))
            self._gripper_step += 1
        else:
            self._gripper_current = self._gripper_target
        return float(np.clip(self._gripper_current, GRIPPER_MIN_PCT, GRIPPER_MAX_PCT))

    def _interpolate(self, a, b, t):
        t_smooth = t * t * (3.0 - 2.0 * t)
        return a + t_smooth * (b - a)

    def _timer_callback(self):
        gripper_pct = self._update_gripper()

        if self._phase == "new_target":
            with self._gen_lock:
                q = self._next_q
                busy = self._generating
            if q is not None:
                with self._gen_lock: self._next_q = None
                self._q_target = q
                self._interp_step = 0
                self._phase = "move"
            elif not busy:
                self._start_generating()
            
            cmd = self._q_current.copy()
            cmd[SO101_DOF - 1] = gripper_pct
            self._publish(cmd)
            return

        if self._phase == "move":
            t = min(self._interp_step / INTERPOLATION_STEPS, 1.0)
            command = self._interpolate(self._q_current, self._q_target, t)
            command[SO101_DOF - 1] = gripper_pct 
            self._publish(command)
            self._interp_step += 1
            if self._interp_step > INTERPOLATION_STEPS:
                self._q_current = self._q_target.copy()
                self._q_current[SO101_DOF - 1] = gripper_pct
                self._hold_step = 0
                self._phase = "hold"
                self._start_generating()
            return

        if self._phase == "hold":
            cmd = self._q_current.copy()
            cmd[SO101_DOF - 1] = gripper_pct
            self._publish(cmd)
            self._hold_step += 1
            if self._hold_step >= WAYPOINT_HOLD_STEPS:
                self._phase = "new_target"
            return

    def _publish(self, q_pct):
        msg = Float64MultiArray()
        msg.data = [float(v) for v in q_pct]
        self.pub_joints.publish(msg)

def main():
    rclpy.init()
    try: rclpy.spin(RandomEEMover())
    except KeyboardInterrupt: pass
    finally: rclpy.shutdown()

if __name__ == "__main__":
    main()

