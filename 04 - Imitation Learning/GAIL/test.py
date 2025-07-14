import gymnasium as gym
from environment import GAILWrapper
from discriminator import Discriminator
from stable_baselines3 import PPO

if __name__ == "__main__":
    disc = Discriminator (3,3)
    env = gym.make("CarRacing-v3")
    w_env = GAILWrapper(env,disc)
    state , info = w_env.reset()
    model = PPO("MlpPolicy",w_env)
    

    
    

    