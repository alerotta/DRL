import numpy as np
import glob
import os 
from environment import EnvWrapper
from discriminator import Discriminator
import gymnasium as gym

if __name__ == "__main__":

    files =  glob.glob(os.path.join("trajectories" , "*npz"))

    data = np.load(files[0])
    states = data["states"]
    actions = data["actions"]

    print(f"len: {len(actions)}, first action: {actions[0]}, shape: states {states[0].shape} actions {actions[0].shape}")

    disc = Discriminator()
    real_env  = gym.make("CarRacing-v3", render_mode = "human")

    w_env = EnvWrapper(real_env,disc)
    action = w_env.action_space.sample()
    w_env.step(action)