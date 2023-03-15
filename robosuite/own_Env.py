from robosuite.models import MujocoWorldBase
from robosuite.models.robots import Maholo
from robosuite.models.robots import IIWA
from robosuite.models.grippers import gripper_factory
from robosuite.models.arenas import TableArena
from robosuite.models.objects import BallObject
from robosuite.utils.mjcf_utils import new_joint
import mujoco
import mujoco_viewer

# Step 1: Creating the world.
world = MujocoWorldBase()

# Step 2: Creating the robot, add a gripper.
mujoco_robot = Maholo()
# mujoco_robot.contact_geom_rgba([0.5, 0.5, 0.5, 1.0])
# gripper = gripper_factory('PandaGripper')
# mujoco_robot.add_gripper(gripper)
mujoco_robot.set_base_xpos([0, 0, 0])
world.merge(mujoco_robot)

# Step 3: Creating the table.
mujoco_arena = TableArena()
mujoco_arena.set_origin([0.8, 0, 0])
world.merge(mujoco_arena)

# Step 4: Adding the object.
sphere = BallObject(
    name="sphere",
    size=[0.04],
    rgba=[0, 0.5, 0.5, 1]).get_obj()
sphere.set('pos', '0.8 0 0.8')
world.worldbody.append(sphere)

# Step 5: Running Simulation.
model = world.get_model(mode="mujoco")
data = mujoco.MjData(model)

# create the viewer object
viewer = mujoco_viewer.MujocoViewer(model, data)

# simulate and render
for _ in range(100000):
    mujoco.mj_step(model, data)
    viewer.render()

# close
viewer.close()
