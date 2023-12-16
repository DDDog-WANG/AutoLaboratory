import mujoco, mujoco_py, mujoco_viewer, mujoco.viewer

ubuntu_root = "/home/wang/Desktop/robosuite/models/assets"
mac_root = "/Users/Alpaca/miniforge3/envs/mujoco_env/lib/python3.10/site-packages/robosuite/models/assets"
mac_ubuntu_root ="/home/parallels/.local/lib/python3.10/site-packages/robosuite/models/assets"

robot = "/robots/maholo/robot.xml"
gripper = "/grippers/maholo_gripper_l.xml"
arena = "/arenas/laboratory_arena.xml"
obj = "/objects/1.5ml_tube.xml"

model_path = mac_root + robot

# 1. mujoco.viewer
model = mujoco.MjModel.from_xml_path(model_path)
data = mujoco.MjData(model)
viewer = mujoco.viewer.launch(model, data)

# # 2. mujoco_viewer
# model = mujoco.MjModel.from_xml_path(model_path)
# data = mujoco.MjData(model)
# viewer = mujoco_viewer.MujocoViewer(model, data)
# while True:
#     mujoco.mj_step(model, data)
#     viewer.render()
# viewer.close()

# # 3. mujoco_py
# model = mujoco_py.load_model_from_path(model_path)
# sim = mujoco_py.MjSim(model)
# viewer = mujoco_py.MjViewer(sim)
# while True:
#     sim.step()
#     viewer.render()
# viewer.close()