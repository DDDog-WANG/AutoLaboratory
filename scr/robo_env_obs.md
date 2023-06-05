WORKDIR: /home/acd13264yb/DDDog/AutoLaboratory/scr
SCRIPT: /home/acd13264yb/DDDog/AutoLaboratory/scr/rl_lift_test.py
SAVETO: /home/acd13264yb/DDDog/AutoLaboratory/scr/models/SAC
🎃 robosuite.env
type(env):  <class 'robosuite.environments.manipulation.lift.Lift'>
dir(env):  ['__class__', '__delattr__', '__dict__', '__dir__', '__doc__', '__eq__', '__format__', '__ge__', '__getattribute__', '__gt__', '__hash__', '__init__', '__init_subclass__', '__le__', '__lt__', '__module__', '__ne__', '__new__', '__reduce__', '__reduce_ex__', '__repr__', '__setattr__', '__sizeof__', '__str__', '__subclasshook__', '__weakref__', '_action_dim', '_check_grasp', '_check_robot_configuration', '_check_success', '_create_camera_sensors', '_create_segementation_sensor', '_destroy_sim', '_destroy_viewer', '_eef_xmat', '_eef_xpos', '_eef_xquat', '_get_observations', '_gripper_to_target', '_initialize_sim', '_input2list', '_load_model', '_load_robots', '_obs_cache', '_observables', '_post_action', '_pre_action', '_reset_internal', '_setup_observables', '_setup_references', '_update_observables', '_visualizations', '_visualize_gripper_to_target', '_xml_processor', 'action_dim', 'action_spec', 'active_observables', 'add_observable', 'camera_depths', 'camera_heights', 'camera_names', 'camera_segmentations', 'camera_widths', 'check_contact', 'clear_objects', 'close', 'close_renderer', 'control_freq', 'control_timestep', 'cube', 'cube_body_id', 'cur_time', 'deterministic_reset', 'done', 'edit_model_xml', 'enabled_observables', 'env_configuration', 'get_contacts', 'get_pixel_obs', 'hard_reset', 'has_offscreen_renderer', 'has_renderer', 'horizon', 'ignore_done', 'initialize_renderer', 'initialize_time', 'model', 'model_timestep', 'modify_observable', 'num_cameras', 'num_robots', 'observation_modalities', 'observation_names', 'observation_spec', 'placement_initializer', 'render', 'render_camera', 'render_collision_mesh', 'render_gpu_device_id', 'render_visual_mesh', 'renderer', 'renderer_config', 'reset', 'reset_from_xml_string', 'reward', 'reward_scale', 'reward_shaping', 'robot_configs', 'robot_names', 'robots', 'set_camera_pos_quat', 'set_xml_processor', 'sim', 'sim_state_initial', 'step', 'table_friction', 'table_full_size', 'table_offset', 'timestep', 'use_camera_obs', 'use_object_obs', 'viewer', 'viewer_get_obs', 'visualize']
env:  <robosuite.environments.manipulation.lift.Lift object at 0x1538bd043eb0>
🌼 [obs] robosuite.env.reset()
type(obs):  <class 'collections.OrderedDict'>
dir(obs):  ['__class__', '__class_getitem__', '__contains__', '__delattr__', '__delitem__', '__dict__', '__dir__', '__doc__', '__eq__', '__format__', '__ge__', '__getattribute__', '__getitem__', '__gt__', '__hash__', '__init__', '__init_subclass__', '__ior__', '__iter__', '__le__', '__len__', '__lt__', '__ne__', '__new__', '__or__', '__reduce__', '__reduce_ex__', '__repr__', '__reversed__', '__ror__', '__setattr__', '__setitem__', '__sizeof__', '__str__', '__subclasshook__', 'clear', 'copy', 'fromkeys', 'get', 'items', 'keys', 'move_to_end', 'pop', 'popitem', 'setdefault', 'update', 'values']
Key: robot0_joint_pos_cos, 
Key: robot0_joint_pos_sin, Value.shape: (7,)
Key: robot0_joint_vel, Value.shape: (7,)
Key: robot0_eef_pos, Value.shape: (3,)
Key: robot0_eef_quat, Value.shape: (4,)
Key: robot0_gripper_qpos, Value.shape: (2,)
Key: robot0_gripper_qvel, Value.shape: (2,)
Key: cube_pos, Value.shape: (3,)
Key: cube_quat, Value.shape: (4,)
Key: gripper_to_cube_pos, Value.shape: (3,)
Key: robot0_proprio-state, Value.shape: (32,)
Key: object-state, Value.shape: (10,)
obs:  OrderedDict([
  ('robot0_joint_pos_cos', array([ 0.99830237,  0.98308307,  0.99998386, -0.85621743,  0.99986495, -0.97954039,  0.71751008])), 
  ('robot0_joint_pos_sin', array([-0.05824407,  0.18316026,  0.00568087, -0.51661564,  0.01643422, 0.20124765,  0.69654812])), 
  ('robot0_joint_vel', array([0., 0., 0., 0., 0., 0., 0.])), 
  ('robot0_eef_pos', array([-0.09108742, -0.02524383,  1.02259388])), 
  ('robot0_eef_quat', array([ 9.96599641e-01, -2.65647862e-02,  7.79964558e-02, -1.43227900e-04])), 
  ('robot0_gripper_qpos', array([ 0.020833, -0.020833])), 
  ('robot0_gripper_qvel', array([0., 0.])), 
  ('cube_pos', array([0.00618791, 0.01069681, 0.83024351])), 
  ('cube_quat', array([ 0.        ,  0.        ,  0.89747885, -0.44105748])), 
  ('gripper_to_cube_pos', array([-0.09727532, -0.03594064,  0.19235038])), 
  ('robot0_proprio-state', array([9.98302373e-01,  9.83083069e-01,  9.99983864e-01, -8.56217427e-01,
                                  9.99864949e-01, -9.79540394e-01,  7.17510085e-01, -5.82440749e-02,
                                  1.83160258e-01,  5.68086585e-03, -5.16615638e-01,  1.64342162e-02,
                                  2.01247651e-01,  6.96548116e-01,  0.00000000e+00,  0.00000000e+00,
                                  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
                                  0.00000000e+00, -9.10874152e-02, -2.52438304e-02,  1.02259388e+00,
                                  9.96599641e-01, -2.65647862e-02,  7.79964558e-02, -1.43227900e-04,
                                  2.08330000e-02, -2.08330000e-02,  0.00000000e+00,  0.00000000e+00])), 
  ('object-state', array([ 0.00618791,  0.01069681,  0.83024351,  0.        ,  0.        ,
                           0.89747885, -0.44105748, -0.09727532, -0.03594064,  0.19235038]))])

🎃 GymWrapper(env)
type(env):  <class 'robosuite.wrappers.gym_wrapper.GymWrapper'>
dir(env):  ['__class__', '__delattr__', '__dict__', '__dir__', '__doc__', '__enter__', '__eq__', '__exit__', '__format__', '__ge__', '__getattr__', '__getattribute__', '__gt__', '__hash__', '__init__', '__init_subclass__', '__le__', '__lt__', '__module__', '__ne__', '__new__', '__reduce__', '__reduce_ex__', '__repr__', '__setattr__', '__sizeof__', '__str__', '__subclasshook__', '__weakref__', '_flatten_obs', '_warn_double_wrap', 'action_dim', 'action_space', 'action_spec', 'class_name', 'close', 'compute_reward', 'env', 'keys', 'metadata', 'modality_dims', 'name', 'obs_dim', 'observation_space', 'observation_spec', 'render', 'reset', 'reward_range', 'seed', 'spec', 'step', 'unwrapped']
env:  <GymWrapper instance>
🌼 [obs] GymWrapper(env).reset()
type(obs):  <class 'numpy.ndarray'>
dir(obs):  ['T', '__abs__', '__add__', '__and__', '__array__', '__array_finalize__', '__array_function__', '__array_interface__', '__array_prepare__', '__array_priority__', '__array_struct__', '__array_ufunc__', '__array_wrap__', '__bool__', '__class__', '__class_getitem__', '__complex__', '__contains__', '__copy__', '__deepcopy__', '__delattr__', '__delitem__', '__dir__', '__divmod__', '__dlpack__', '__dlpack_device__', '__doc__', '__eq__', '__float__', '__floordiv__', '__format__', '__ge__', '__getattribute__', '__getitem__', '__gt__', '__hash__', '__iadd__', '__iand__', '__ifloordiv__', '__ilshift__', '__imatmul__', '__imod__', '__imul__', '__index__', '__init__', '__init_subclass__', '__int__', '__invert__', '__ior__', '__ipow__', '__irshift__', '__isub__', '__iter__', '__itruediv__', '__ixor__', '__le__', '__len__', '__lshift__', '__lt__', '__matmul__', '__mod__', '__mul__', '__ne__', '__neg__', '__new__', '__or__', '__pos__', '__pow__', '__radd__', '__rand__', '__rdivmod__', '__reduce__', '__reduce_ex__', '__repr__', '__rfloordiv__', '__rlshift__', '__rmatmul__', '__rmod__', '__rmul__', '__ror__', '__rpow__', '__rrshift__', '__rshift__', '__rsub__', '__rtruediv__', '__rxor__', '__setattr__', '__setitem__', '__setstate__', '__sizeof__', '__str__', '__sub__', '__subclasshook__', '__truediv__', '__xor__', 'all', 'any', 'argmax', 'argmin', 'argpartition', 'argsort', 'astype', 'base', 'byteswap', 'choose', 'clip', 'compress', 'conj', 'conjugate', 'copy', 'ctypes', 'cumprod', 'cumsum', 'data', 'diagonal', 'dot', 'dtype', 'dump', 'dumps', 'fill', 'flags', 'flat', 'flatten', 'getfield', 'imag', 'item', 'itemset', 'itemsize', 'max', 'mean', 'min', 'nbytes', 'ndim', 'newbyteorder', 'nonzero', 'partition', 'prod', 'ptp', 'put', 'ravel', 'real', 'repeat', 'reshape', 'resize', 'round', 'searchsorted', 'setfield', 'setflags', 'shape', 'size', 'sort', 'squeeze', 'std', 'strides', 'sum', 'swapaxes', 'take', 'tobytes', 'tofile', 'tolist', 'tostring', 'trace', 'transpose', 'var', 'view']
obs.shape:  (42,)
obs:  [-0.00804811  0.01043878  0.83159615  0.          0.          0.08601619
        0.99629374 -0.09307489 -0.03045748  0.17614035  0.99976036  0.97788616
        0.99981434 -0.8635704   0.99997341 -0.98294843  0.70917878 -0.02189123
        0.20913791 -0.01926871 -0.50422829  0.00729256  0.18388146  0.70502869
        0.          0.          0.          0.          0.          0.
        0.         -0.101123   -0.02001869  1.0077365   0.99754922 -0.02200976
        0.06636658  0.00256947  0.020833   -0.020833    0.          0.        ]

🎃 TimeFeatureWrapper(env)
type(env):  <class 'sb3_contrib.common.wrappers.time_feature.TimeFeatureWrapper'>
dir(env):  ['__class__', '__delattr__', '__dict__', '__dir__', '__doc__', '__enter__', '__eq__', '__exit__', '__format__', '__ge__', '__getattr__', '__getattribute__', '__gt__', '__hash__', '__init__', '__init_subclass__', '__le__', '__lt__', '__module__', '__ne__', '__new__', '__reduce__', '__reduce_ex__', '__repr__', '__setattr__', '__sizeof__', '__str__', '__subclasshook__', '__weakref__', '_action_space', '_current_step', '_get_obs', '_max_steps', '_metadata', '_observation_space', '_reward_range', '_test_mode', 'action_space', 'class_name', 'close', 'compute_reward', 'dtype', 'env', 'metadata', 'observation_space', 'render', 'reset', 'reward_range', 'seed', 'spec', 'step', 'unwrapped']
env:  <TimeFeatureWrapper<GymWrapper instance>>
🌼 [obs] TimeFeatureWrapper(env).reset()
type(obs):  <class 'numpy.ndarray'>
dir(obs):  ['T', '__abs__', '__add__', '__and__', '__array__', '__array_finalize__', '__array_function__', '__array_interface__', '__array_prepare__', '__array_priority__', '__array_struct__', '__array_ufunc__', '__array_wrap__', '__bool__', '__class__', '__class_getitem__', '__complex__', '__contains__', '__copy__', '__deepcopy__', '__delattr__', '__delitem__', '__dir__', '__divmod__', '__dlpack__', '__dlpack_device__', '__doc__', '__eq__', '__float__', '__floordiv__', '__format__', '__ge__', '__getattribute__', '__getitem__', '__gt__', '__hash__', '__iadd__', '__iand__', '__ifloordiv__', '__ilshift__', '__imatmul__', '__imod__', '__imul__', '__index__', '__init__', '__init_subclass__', '__int__', '__invert__', '__ior__', '__ipow__', '__irshift__', '__isub__', '__iter__', '__itruediv__', '__ixor__', '__le__', '__len__', '__lshift__', '__lt__', '__matmul__', '__mod__', '__mul__', '__ne__', '__neg__', '__new__', '__or__', '__pos__', '__pow__', '__radd__', '__rand__', '__rdivmod__', '__reduce__', '__reduce_ex__', '__repr__', '__rfloordiv__', '__rlshift__', '__rmatmul__', '__rmod__', '__rmul__', '__ror__', '__rpow__', '__rrshift__', '__rshift__', '__rsub__', '__rtruediv__', '__rxor__', '__setattr__', '__setitem__', '__setstate__', '__sizeof__', '__str__', '__sub__', '__subclasshook__', '__truediv__', '__xor__', 'all', 'any', 'argmax', 'argmin', 'argpartition', 'argsort', 'astype', 'base', 'byteswap', 'choose', 'clip', 'compress', 'conj', 'conjugate', 'copy', 'ctypes', 'cumprod', 'cumsum', 'data', 'diagonal', 'dot', 'dtype', 'dump', 'dumps', 'fill', 'flags', 'flat', 'flatten', 'getfield', 'imag', 'item', 'itemset', 'itemsize', 'max', 'mean', 'min', 'nbytes', 'ndim', 'newbyteorder', 'nonzero', 'partition', 'prod', 'ptp', 'put', 'ravel', 'real', 'repeat', 'reshape', 'resize', 'round', 'searchsorted', 'setfield', 'setflags', 'shape', 'size', 'sort', 'squeeze', 'std', 'strides', 'sum', 'swapaxes', 'take', 'tobytes', 'tofile', 'tolist', 'tostring', 'trace', 'transpose', 'var', 'view']
obs.shape:  (43,)
obs:  [ 7.78328696e-04 -2.12326608e-02  8.30677912e-01  0.00000000e+00
        0.00000000e+00  5.47102080e-01  8.37065896e-01 -9.46036035e-02
        2.19163034e-02  1.83021621e-01  9.99709402e-01  9.79048069e-01
        9.99756411e-01 -8.52663520e-01  9.99820156e-01 -9.77063242e-01
        6.96083586e-01 -2.41062676e-02  2.03629266e-01  2.20707653e-02
       -5.22460450e-01 -1.89645901e-02  2.12949341e-01  7.17960752e-01
        0.00000000e+00  0.00000000e+00  0.00000000e+00  0.00000000e+00
        0.00000000e+00  0.00000000e+00  0.00000000e+00 -9.38252748e-02
        6.83642520e-04  1.01369953e+00  9.97882895e-01  2.48898881e-04
        6.50358695e-02 -3.06099030e-05  2.08330000e-02 -2.08330000e-02
        0.00000000e+00  0.00000000e+00  1.00000000e+00]

Wrapping the env with a `Monitor` wrapper
Wrapping the env in a DummyVecEnv.
🔱 001 ['0.1349', '-0.0168', '-0.1002', '-0.1295', '-0.0280', '0.0146', '-0.0487', '-0.1114'] 0.00757
🔱 002 ['0.1348', '-0.0168', '-0.1001', '-0.1294', '-0.0280', '0.0146', '-0.0487', '-0.1113'] 0.00685
🔱 003 ['0.1349', '-0.0170', '-0.1002', '-0.1297', '-0.0280', '0.0146', '-0.0488', '-0.1115'] 0.00618
🔱 004 ['0.1354', '-0.0176', '-0.1008', '-0.1309', '-0.0280', '0.0147', '-0.0491', '-0.1120'] 0.00618
🔱 005 ['0.1357', '-0.0181', '-0.1012', '-0.1319', '-0.0281', '0.0148', '-0.0494', '-0.1124'] 0.00625

🌼 [obs]
type(obs):  <class 'numpy.ndarray'>
dir(obs):  ['T', '__abs__', '__add__', '__and__', '__array__', '__array_finalize__', '__array_function__', '__array_interface__', '__array_prepare__', '__array_priority__', '__array_struct__', '__array_ufunc__', '__array_wrap__', '__bool__', '__class__', '__class_getitem__', '__complex__', '__contains__', '__copy__', '__deepcopy__', '__delattr__', '__delitem__', '__dir__', '__divmod__', '__dlpack__', '__dlpack_device__', '__doc__', '__eq__', '__float__', '__floordiv__', '__format__', '__ge__', '__getattribute__', '__getitem__', '__gt__', '__hash__', '__iadd__', '__iand__', '__ifloordiv__', '__ilshift__', '__imatmul__', '__imod__', '__imul__', '__index__', '__init__', '__init_subclass__', '__int__', '__invert__', '__ior__', '__ipow__', '__irshift__', '__isub__', '__iter__', '__itruediv__', '__ixor__', '__le__', '__len__', '__lshift__', '__lt__', '__matmul__', '__mod__', '__mul__', '__ne__', '__neg__', '__new__', '__or__', '__pos__', '__pow__', '__radd__', '__rand__', '__rdivmod__', '__reduce__', '__reduce_ex__', '__repr__', '__rfloordiv__', '__rlshift__', '__rmatmul__', '__rmod__', '__rmul__', '__ror__', '__rpow__', '__rrshift__', '__rshift__', '__rsub__', '__rtruediv__', '__rxor__', '__setattr__', '__setitem__', '__setstate__', '__sizeof__', '__str__', '__sub__', '__subclasshook__', '__truediv__', '__xor__', 'all', 'any', 'argmax', 'argmin', 'argpartition', 'argsort', 'astype', 'base', 'byteswap', 'choose', 'clip', 'compress', 'conj', 'conjugate', 'copy', 'ctypes', 'cumprod', 'cumsum', 'data', 'diagonal', 'dot', 'dtype', 'dump', 'dumps', 'fill', 'flags', 'flat', 'flatten', 'getfield', 'imag', 'item', 'itemset', 'itemsize', 'max', 'mean', 'min', 'nbytes', 'ndim', 'newbyteorder', 'nonzero', 'partition', 'prod', 'ptp', 'put', 'ravel', 'real', 'repeat', 'reshape', 'resize', 'round', 'searchsorted', 'setfield', 'setflags', 'shape', 'size', 'sort', 'squeeze', 'std', 'strides', 'sum', 'swapaxes', 'take', 'tobytes', 'tofile', 'tolist', 'tostring', 'trace', 'transpose', 'var', 'view']
obs.shape:  (43,)
obs:  [ 7.78328696e-04 -2.12326608e-02  8.20430804e-01  2.93264929e-17
        1.97428551e-17  5.47102080e-01  8.37065896e-01 -9.81834468e-02
        2.09673269e-02  1.91435068e-01  9.99809470e-01  9.79273105e-01
        9.99874147e-01 -8.56493220e-01  9.99815401e-01 -9.77067039e-01
        6.96093072e-01 -1.95198325e-02  2.02544281e-01  1.58647460e-02
       -5.16158274e-01 -1.92136510e-02  2.12931915e-01  7.17951555e-01
        2.43017181e-02 -5.70318489e-03 -3.18937718e-02 -3.72648592e-02
       -2.44135080e-03  5.12975286e-05 -4.02575895e-05 -9.74051181e-02
       -2.65333899e-04  1.01186587e+00  9.98079865e-01 -3.09160101e-04
        6.19357418e-02  6.71853712e-04  3.81304263e-02 -3.81359140e-02
        1.01988945e-02 -1.01972050e-02  9.80000019e-01]

