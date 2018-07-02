import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env

class MonopodEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        mujoco_env.MujocoEnv.__init__(self, 'monopod.xml', 4) # 4 is frameskip
        utils.EzPickle.__init__(self)

    def step(self, a):
        ctrl_cost_coeff = 0.0001
        xposbefore = self.sim.data.qpos[0]
        self.do_simulation(a, self.frame_skip)
        xposafter = self.sim.data.qpos[0]
        reward_fwd = 10*(xposafter - xposbefore) / self.dt
        reward_ctrl = - ctrl_cost_coeff * np.square(a).sum()
        # reward_height = (0.7 - self.sim.data.qpos[1]) ** 2
        reward = reward_fwd + reward_ctrl
        ob = self._get_obs()

        done = False
        zpos = self.sim.data.qpos[1]    # see xml file

        # print(zpos)
        if zpos < 0.4:
            reward = reward - 50    # reward for falling
            done = True

        if not done:
            reward += 1

        return ob, reward, done, dict(reward_fwd=reward_fwd, reward_ctrl=reward_ctrl)

    def _get_obs(self):
        qpos = self.sim.data.qpos
        qvel = self.sim.data.qvel
        return np.concatenate([qpos.flat[2:], qvel.flat])

    def reset_model(self):
        self.set_state(
            self.init_qpos + self.np_random.uniform(low=-.1, high=.1, size=self.model.nq),
            self.init_qvel + self.np_random.uniform(low=-.1, high=.1, size=self.model.nv)
        )
        return self._get_obs()
