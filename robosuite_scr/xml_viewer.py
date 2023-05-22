import mujoco
import mujoco_viewer
from robosuite.models.robots import Maholo

maholo_robot="/media/psf/Home/Desktop/Github/Maholo/robosuite_dir/models/assets/robots/baxter/robot.xml"
maholo_gripper="/media/psf/Home/Desktop/Github/Maholo/robosuite_dir/models/assets/grippers/maholo_gripper_r.xml"
gripper="/media/psf/Home/Desktop/Github/Maholo/robosuite_dir/models/assets/grippers/robotiq_gripper_140.xml"

# maholo_robot="/media/psf/Home/Desktop/Github/Maholo/AutoLabRob/robosuite/robots/maholo/robot_b.xml"
model = mujoco.MjModel.from_xml_path(maholo_robot)
mujoco_robot = Maholo()
data = mujoco.MjData(model)

# create the viewer object
viewer = mujoco_viewer.MujocoViewer(model, data)

# simulate and render
for _ in range(100000):
    if viewer.is_alive:
        mujoco.mj_step(model, data)
        viewer.render()
    else:
        break

# close
viewer.close()