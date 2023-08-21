import mujoco
import mujoco_viewer
from robosuite.models.robots import Maholo

ubuntu_dir = "/home/wang/Desktop/robosuite"
mac_dir = "/media/psf/Home/Desktop/Github/Maholo/robosuite_dir"
mac_ubuntu_dir ="/home/parallels/.local/lib/python3.10/site-packages/robosuite"
maholo_robot = "/models/assets/robots/maholo/robot.xml"
gripper = "/models/assets/grippers/maholo_gripper_l.xml"
room = "/models/assets/arenas/laboratory_arena.xml"
object = "/models/assets/objects/1.5ml tube main.xml"
obj = "/models/assets/objects/1.5ml_tube.xml"
model = mujoco.MjModel.from_xml_path(mac_ubuntu_dir + maholo_robot)
mujoco_robot = Maholo()
data = mujoco.MjData(model)

# create the viewer object
viewer = mujoco_viewer.MujocoViewer(model, data)

# simulate and render
for _ in range(5000):
    if viewer.is_alive:
        mujoco.mj_step(model, data)
        viewer.render()
    else:
        break

# close
viewer.close()
