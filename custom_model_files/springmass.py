import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env

class SpringmassEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        mujoco_env.MujocoEnv.__init__(self, 'springmass.xml', 1) # 4 is frameskip
        utils.EzPickle.__init__(self)

    def step(self, a):
        self.do_simulation(a, self.frame_skip)
        box1_zpos = self.sim.data.qpos[0]    # upper box
        box2_zpos = self.sim.data.qpos[1]

        reward = -(8.0 - box1_zpos)**2
        ob = self._get_obs()
        done = False

        # print("pos = " + str(box1_zpos) + "    reward = " + str(reward))
        # print("box1 " + str(box1_zpos) + "    box2  " + str(box2_zpos))

        return ob, reward, False, dict(reward_fwd=0, reward_ctrl=0)

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
