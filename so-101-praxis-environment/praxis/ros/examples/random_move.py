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

# Gripper randomisation — pick a new random target every N timer ticks
GRIPPER_CHANGE_TICKS = 60   # ~1.5 s at 40 Hz
GRIPPER_INTERP_TICKS = 40   # ~1.0 s to open/close
GRIPPER_MIN_PCT = 0.0        # fully closed
GRIPPER_MAX_PCT = 100.0      # fully open


class RandomEEMover(Node):
    def __init__(self):
        super().__init__(node_name="random_ee_mover")

        # Make it not print anything
        self.get_logger().set_level(rclpy.logging.LoggingSeverity.FATAL)

        self.pub_joints = self.create_publisher(
            msg_type=Float64MultiArray,
            topic="so101real/joint_command",
            qos_profile=10,
        )

        self._init_mujoco()

        self._q_current = self._neutral_percentage()
        self.get_logger().info(
            f"Neutral joint config (%%): {np.round(self._q_current, 1).tolist()}"
        )

        # Arm interpolation state
        self._q_target: np.ndarray | None = None
        self._interp_step = 0
        self._hold_step   = 0
        self._phase       = "new_target"

        # Background waypoint generation (arm joints only, DOF 0-4)
        self._next_q: np.ndarray | None = None
        self._generating = False
        self._gen_lock   = threading.Lock()

        # Gripper state — runs independently of the arm
        self._gripper_current = 50.0
        self._gripper_target  = 50.0
        self._gripper_step    = 0
        self._gripper_change_countdown = GRIPPER_CHANGE_TICKS

        self.timer = self.create_timer(0.025, self._timer_callback)
        self.get_logger().info("RandomEEMover ready — publishing to so101real/joint_command")

    # ------------------------------------------------------------------
    # MuJoCo
    # ------------------------------------------------------------------
    def _init_mujoco(self):
        self.get_logger().info(f"Loading MuJoCo model: {SO101_XML_PATH}")
        if not os.path.exists(SO101_XML_PATH):
            raise FileNotFoundError(f"MuJoCo XML not found: {SO101_XML_PATH}")

        self._spec     = MjSpec.from_file(SO101_XML_PATH)
        self._model    = self._spec.compile()
        self._data     = MjData(self._model)
        self._data_gen = MjData(self._model)

        self._ee_site_id = mujoco.mj_name2id(
            self._model, mujoco.mjtObj.mjOBJ_SITE, SO101_EE_SITE_NAME
        )
        assert self._ee_site_id >= 0, f"Site '{SO101_EE_SITE_NAME}' not found"

        self._lower_rad = self._model.jnt_range[:SO101_DOF, 0].copy()
        self._upper_rad = self._model.jnt_range[:SO101_DOF, 1].copy()
        self.get_logger().info("MuJoCo model loaded OK.")

    def _neutral_percentage(self) -> np.ndarray:
        q = np.zeros(SO101_DOF)
        q[SO101_DOF - 1] = 50.0  # gripper half-open
        return q

    # ------------------------------------------------------------------
    # Unit conversions
    # ------------------------------------------------------------------
    def _percentage_to_radians(self, q_pct: np.ndarray) -> np.ndarray:
        q_rad = np.zeros(SO101_DOF)
        for i in range(SO101_DOF - 1):
            q_rad[i] = (
                self._lower_rad[i]
                + ((q_pct[i] + 100) / 200.0)
                * (self._upper_rad[i] - self._lower_rad[i])
            )
        g = SO101_DOF - 1
        q_rad[g] = (
            self._lower_rad[g]
            + (q_pct[g] / 100.0)
            * (self._upper_rad[g] - self._lower_rad[g])
        )
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

    # ------------------------------------------------------------------
    # Safety checks
    # ------------------------------------------------------------------
    def _has_self_collision(self, mjdata: MjData) -> bool:
        for i in range(mjdata.ncon):
            if mjdata.contact[i].dist < -SELF_COLLISION_MARGIN:
                return True
        return False

    def _ee_above_table(self, mjdata: MjData) -> bool:
        return float(mjdata.site(self._ee_site_id).xpos[2]) >= TABLE_Z_MIN

    # ------------------------------------------------------------------
    # Background arm waypoint generation
    # ------------------------------------------------------------------
    def _start_generating(self):
        with self._gen_lock:
            if self._generating:
                return
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

                if self._has_self_collision(self._data_gen):
                    continue
                if not self._ee_above_table(self._data_gen):
                    continue

                dist = np.linalg.norm(
                    self._data_gen.site(self._ee_site_id).xpos - target_pos
                )
                if dist < best_dist:
                    best_dist  = dist
                    best_q_rad = q_rad.copy()

            if best_q_rad is not None and best_dist < 0.10:
                result = self._radians_to_percentage(best_q_rad)
                # Note: gripper index is left in result but overwritten each tick
                # by the independent gripper controller — value here doesn't matter.
                break

        with self._gen_lock:
            self._next_q     = result
            self._generating = False

        if result is not None:
            q_rad = self._percentage_to_radians(result)
            self._data.qpos[:SO101_DOF] = q_rad
            mujoco.mj_forward(self._model, self._data)
            pos = self._data.site(self._ee_site_id).xpos
            self.get_logger().info(
                f"Waypoint ready: EE=({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f})"
            )
        else:
            self.get_logger().warn("Could not find a valid waypoint — will retry.")

    # ------------------------------------------------------------------
    # Gripper controller (independent of arm, called every timer tick)
    # ------------------------------------------------------------------
    def _update_gripper(self) -> float:
        """
        Smoothly interpolates the gripper toward its target, and picks a new
        random target whenever the countdown expires.
        Returns the current gripper percentage to embed in the joint command.
        """
        # Count down to next random gripper target
        self._gripper_change_countdown -= 1
        if self._gripper_change_countdown <= 0:
            self._gripper_target = float(
                np.random.uniform(GRIPPER_MIN_PCT, GRIPPER_MAX_PCT)
            )
            self._gripper_step = 0
            self._gripper_change_countdown = GRIPPER_CHANGE_TICKS
            self.get_logger().info(
                f"Gripper target: {self._gripper_target:.1f}%"
            )

        # Smooth interpolation toward target
        if self._gripper_step < GRIPPER_INTERP_TICKS:
            t = self._gripper_step / GRIPPER_INTERP_TICKS
            t_smooth = t * t * (3.0 - 2.0 * t)
            self._gripper_current = (
                self._gripper_current
                + t_smooth * (self._gripper_target - self._gripper_current)
            )
            self._gripper_step += 1
        else:
            self._gripper_current = self._gripper_target

        return float(np.clip(self._gripper_current, GRIPPER_MIN_PCT, GRIPPER_MAX_PCT))

    # ------------------------------------------------------------------
    # Interpolation (arm joints only)
    # ------------------------------------------------------------------
    @staticmethod
    def _smooth_step(t: float) -> float:
        return t * t * (3.0 - 2.0 * t)

    def _interpolate(self, a: np.ndarray, b: np.ndarray, t: float) -> np.ndarray:
        return a + self._smooth_step(t) * (b - a)

    # ------------------------------------------------------------------
    # Timer callback (40 Hz)
    # ------------------------------------------------------------------
    def _timer_callback(self):
        # Always update the gripper independently
        gripper_pct = self._update_gripper()

        if self._phase == "new_target":
            with self._gen_lock:
                q    = self._next_q
                busy = self._generating

            if q is not None:
                with self._gen_lock:
                    self._next_q = None
                self._q_target    = q
                self._interp_step = 0
                self._phase       = "move"
            elif not busy:
                self._start_generating()

            cmd = self._q_current.copy()
            cmd[SO101_DOF - 1] = gripper_pct
            self._publish(cmd)
            return

        if self._phase == "move":
            t       = min(self._interp_step / INTERPOLATION_STEPS, 1.0)
            command = self._interpolate(self._q_current, self._q_target, t)
            command[SO101_DOF - 1] = gripper_pct   # overwrite with live gripper value
            self._publish(command)
            self._interp_step += 1
            if self._interp_step > INTERPOLATION_STEPS:
                self._q_current            = self._q_target.copy()
                self._q_current[SO101_DOF - 1] = gripper_pct
                self._hold_step            = 0
                self._phase                = "hold"
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

    def _publish(self, q_pct: np.ndarray):
        msg      = Float64MultiArray()
        msg.data = [float(v) for v in q_pct]
        self.pub_joints.publish(msg)


# ======================================================================
def main():
    rclpy.init()
    try:
        node = RandomEEMover()
    except Exception as e:
        print(f"[FATAL] {e}")
        import traceback; traceback.print_exc()
        rclpy.shutdown()
        return

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()


