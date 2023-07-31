import mujoco
import mujoco_viewer
from robosuite.models.robots import Maholo

maholo_robot="/media/psf/Home/Desktop/Github/Maholo/robosuite_dir/models/assets/robots/baxter/robot.xml"
maholo_gripper="/home/wang/Desktop/robosuite/models/assets/grippers/maholo_gripper_l.xml"
gripper="/media/psf/Home/Desktop/Github/Maholo/robosuite_dir/models/assets/grippers/robotiq_gripper_140.xml"
room="/media/psf/Home/Desktop/Github/Maholo/robosuite_ubuntu/models/assets/arenas/laboratory_arena.xml"
object="/media/psf/Home/Desktop/Github/Maholo/robosuite_ubuntu/models/assets/objects/1.5ml tube main.xml"
obj="/home/wang/Desktop/robosuite/models/assets/objects/1.5ml_tube.xml"
model = mujoco.MjModel.from_xml_path(maholo_gripper)
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
