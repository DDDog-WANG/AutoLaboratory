import numpy as np

def step(self, action):
	"""
	Takes a step in simulation with control command @action.
	Args:
		action (np.array): Action to execute within the environment
	Returns:
		4-tuple:
			- (OrderedDict) observations from the environment 
			- (float) reward from the environment
			- (bool) whether the current episode is completed or not
			- (dict) misc information
	Raises:
		ValueError: [Steps past episode termination]
	"""
	if self.done:
		raise ValueError("executing action in terminated episode")
	self.timestep += 1
	policy_step = True
	# Loop through the simulation at the model timestep rate until we re ready to take the next policy step
	# as defined by the control frequency specified at the environment level)
	for i in range(int(self.control_timestep / self.model_timestep)):
		self.sim. forward()
		self._pre_action(action, policy_step) 
		self.sim.step()
		self._update_observables()
		policy_step = False
	# Note: this is done all at once to avoid floating point inaccuracies
	self.cur_time += self.control_timestep 
	reward, done, info = self._post_action(action)
	return self._get_observations, reward, done, info


	# @sensor(modality=modality)
	# def angle(obs_cache):
	# 	t, d, cos = self._compute_orientation()
	# 	obs_cache["+"] = t
	# 	obs_cache["d"] = d
	# 	return cos

	# @sensor(modality=modality)
	# def t(obs_cache):
	# 	return obs_cache["t"] if "t" in obs_cache else 0.0

	# @sensor (modality=modality)
	# def d(obs_cache):
	# 	return obs_cache["d"] if "d" in obs_cache else 0.0

	# 	sensors = hole_pos, hole_quat, peg_to_hole, peg_quat, angle, t, d]
	# 	names = [s._name for s in sensors]
	# 	# Create observables
	# 	for name, s in zip (names, sensors):
	# 		observables[name] = Observable(
	# 		name=name,
	# 		sensor=s,
	# 		sampling_rate=self.control_freq,
	# 		)
	# return observables

def post_action(self, action):
	"""
	Do any housekeeping after taking an action.

	Args:
		action (np. array): Action to execute within the environment

	Returns:
		3-tuple:
			- (float) reward from the environment
			- (bool) whether the current episode is completed or not 
			- (dict) empty dict to be filled with information by
	subclassed method
	"""
	reward = self.reward(action)
	# done if number of elapsed timesteps is greater than horizon
	self.done = (self.timestep >= self.horizon) and not self.ignore_done
	return reward, self.done, {}

def reward(self, action=None):
	"""
	Reward function for the task.

	Sparse un-normalized reward:
		- a discrete reward of 5.0 is provided if the peg is inside the plate's hole
		- Note that we enforce that it's inside at an appropriate angle (cos (theta) > 0.95).
		- Reaching: in [0, 1], to encourage the arms to approach each other
		- Perpendicular Distance: in [0, 1], to encourage the arms to approach each other
		- Parallel Distance: in [0, 1], to encourage the arms to approach each other
		- Alignment: in [0, 1], to encourage having the right orientation between the peg and hole.
		- Placement: in {0, 1], nonzero if the peg is in the hole with a relatively correct alignment
	"""
	reward = 0
	# Right location and angle
	if self._check_success():
		reward = 1.0
	# use a shaping reward
	if self. reward_shaping:
		# Grab relevant values
		t, d, cos = self._compute_orientation()
		# reaching reward
		hole_pos = self.sim.data.body_xpos[self.hole_body_id]
		gripper_site_pos = self.sim.data.body_xpos[self.peg_body_id]
		dist = np.linalg.norm(gripper_site_pos - hole_pos)
		reaching_reward = 1 - np.tanh(1.0 * dist)
		reward += reaching_reward
		# Orientation reward
		reward += 1 - np.tanh(d)
		reward += 1 - np.tanh(np.abs(t))
		reward += cos
	else:
		reward *=5.0
	if self.reward_scale is not None:
		reward *= self.reward_scale / 5.0
	return reward
