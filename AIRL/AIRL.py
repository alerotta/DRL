import Network as net
import gymnasium as gym
from Discriminator import Discriminator
from Generator import PPO 
import torch
import os


            

class AIRL:

    def __init__(self):

        self.environemnt = gym.make("CarRacing-v3")
        self.shared_conv = net.CNNEncoder(self.environemnt.observation_space.shape)
        self.expert_path = "put here trajectory dir path"
        self.discriminator = Discriminator(state_dim=self.environemnt.observation_space.shape,
                                           action_dim=self.environemnt.action_space.shape[0],
                                           expert_path=self.expert_path,
                                           state_encoder=self.shared_conv)
    
        self.generator = PPO(env=self.environemnt,
                            discriminator=self.discriminator,
                            state_encoder=self.shared_conv)

    def train (self, max_episodes = 1000 ,  max_steps_per_episode=500, update_frequency=15):

        all_steps = []
        for episode in range(max_episodes):

            steps = self.generator.play_episode(max_steps_per_episode)
            all_steps.extend(steps)

    
            if (episode + 1) % update_frequency == 0 or episode == max_episodes - 1:
                print(f"Updating policy at episode {episode + 1}")
                self.generator.update_policy(all_steps)
                self.discriminator.update_discriminator(all_steps)

                all_steps = []
        

    def save_model(self, dirpath):
        os.makedirs(dirpath, exist_ok=True)
        torch.save(self.generator.network.state_dict(), f"{dirpath}/policy.pt")
        torch.save(self.discriminator.network.state_dict(), f"{dirpath}/disc.pt")
        print(f"Saved to {dirpath}")

    def load_model(self, dirpath):
        self.generator.network.load_state_dict(torch.load(f"{dirpath}/policy.pt"))
        self.discriminator.network.load_state_dict(torch.load(f"{dirpath}/disc.pt"))
        print(f"Loaded from {dirpath}")


if __name__ == "__main__" :

    airl = AIRL()
    airl.train()
    


                