from robosuite.models import MujocoWorldBase
from robosuite.models import robots
from robosuite.models.grippers import gripper_factory
from robosuite.models import arenas
from robosuite.models import objects
from robosuite.utils.mjcf_utils import new_joint
import mujoco
import mujoco_viewer
import numpy as np

import warnings
warnings.filterwarnings("ignore", category=UserWarning)
# Step 1: Creating the world.
world = MujocoWorldBase()

# Step 2: Creating the aewna.
mujoco_arena = arenas.LaboratoryArena()
mujoco_arena.set_origin([0, 0, 0])
world.merge(mujoco_arena)

# Step 3: Creating the robot.
mujoco_robot = robots.Maholo()
print(mujoco_robot.eef_name)

# Step 4: Creating the gripper.
gripper_right = gripper_factory("MaholoGripper_R", idn=0)
gripper_left = gripper_factory("MaholoGripper_L", idn=1)
gripper = gripper_factory("PandaGripper")
mujoco_robot.add_gripper(gripper_right, arm_name="robot0_right_hand")
mujoco_robot.add_gripper(gripper_left, arm_name="robot0_left_hand")
mujoco_robot.set_base_xpos([-0.620, 0.338, 0.2243])
world.merge(mujoco_robot)

# Step 5: Creating the object.
# object00 = objects.tube1_5mlObject(name="tube1_5ml")
# world.merge(object00)

object01 = objects.P1000Pipette_withtipObject(name="P1000Pipette_withtip").get_obj()
object01.set("pos", "0.5 0.5 0.5")
world.worldbody.append(object01)

# Step 6: Running Simulation.
model = world.get_model(mode="mujoco")
data = mujoco.MjData(model)
# create the viewer object
viewer = mujoco_viewer.MujocoViewer(model, data)
# simulate and render
for _ in range(500):
    mujoco.mj_step(model, data)
    viewer.render()
# close
viewer.close()
