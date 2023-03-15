import mujoco
import mujoco_viewer
from robosuite.models.robots import Maholo

model = mujoco.MjModel.from_xml_path('robot.xml')
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