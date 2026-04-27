lerobot-teleoperate \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACM2 \
    --robot.id=piplup_follower \
    --robot.calibration=/home/Shared/calibration/robots/so101_follower/ \
    --teleop.type=so101_leader \
    --teleop.port=/dev/ttyACM1 \
    --teleop.id=piplup_leader \
    --teleop.calibration_dir=/home/Shared/calibration/teleoperators/so101_leader/ \

