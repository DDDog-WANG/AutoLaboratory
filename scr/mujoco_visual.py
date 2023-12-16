import numpy as np
from robosuite.models import MujocoWorldBase
from robosuite.models import robots, arenas, objects
from robosuite.models.grippers import gripper_factory
import mujoco, mujoco.viewer, mujoco_viewer
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
gripper_right = gripper_factory("MaholoGripper", idn=0)
gripper_left = gripper_factory("MaholoGripper", idn=1)
gripper = gripper_factory("PandaGripper")
mujoco_robot.add_gripper(gripper_right, arm_name="robot0_right_hand")
mujoco_robot.add_gripper(gripper_left, arm_name="robot0_left_hand")
mujoco_robot.set_base_xpos([-0.620, 0.338, 0.2243])
world.merge(mujoco_robot)

# Step 5: Creating the object.
object01 = objects.P1000Pipette_withtipObject(name="P1000Pipette_withtip")
world.merge(object01)
# object01.set_base_xpos([1, 1, 1])
# object00 = objects.tube1_5mlObject(name="tube1_5ml")
# world.merge(object00)

# Step 6: Running Simulation.
model = world.get_model(mode="mujoco")
data = mujoco.MjData(model)
viewer = mujoco.viewer.launch(model, data)