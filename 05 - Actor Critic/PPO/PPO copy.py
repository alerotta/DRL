from Network import PPONetwork
import gymnasium as gym
import numpy as np
import torch

from torch.utils.data import Dataset , DataLoader 
from dataclasses import dataclass
from collections import deque 

@dataclass
class Step :
    state : torch.Tensor
    action : torch.Tensor
    log_prob : torch.Tensor
    value : torch.Tensor
    reward : torch.Tensor
    done: int
    advantage : torch.Tensor
    ret : torch.Tensor 

class PPODataset (Dataset) :

    def __init__(self,steps) :
        self.steps = steps
        

    def __len__(self):
        return len(self.steps)
    
    def __getitem__(self, index):
        step = self.steps[index]
        return {
            'state': step.state,
            'action': step.action,
            'log_prob': step.log_prob,
            'value': step.value,
            'reward': step.reward,
            'done': step.done,
            'advantage': step.advantage,
            'ret': step.ret
        }




class PPO:

    def __init__(self,
                hidden_size = 256,
                gamma  = 0.99,
                lam  = 0.95,
                clip_epsilon  = 0.2,
                value_coef = 0.5,
                entropy_coef = 0.01):
        
        self.gamma = gamma
        self.lam = lam
        self.clip_epsilon = clip_epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        
        self.env = gym.make("CarRacing-v3" )
        self.net = PPONetwork(self.env.observation_space.shape , self.env.action_space.shape[0] , hidden_size)

    
    def play_episode(self, max_len):

        steps = []
        state, _ = self.env.reset()
        state_deuqe_tensor = deque(maxlen=4)

        for _ in range(4):
            state_deuqe_tensor.append(torch.tensor(state, dtype=torch.float32).permute(2,0,1) / 255.0)


        for i in range(max_len):

            tensor = torch.cat(list(state_deuqe_tensor), dim=0).unsqueeze(0)
            
            with torch.no_grad():
                action, log_prob, value = self.net.sample_action(tensor)
            
            next_state, reward, is_terminated, is_truncated, _ = self.env.step(action.squeeze(0).numpy())
            done = 1 if (is_terminated or is_truncated or i == max_len - 1) else 0
        
            s = Step(
                state=tensor.squeeze(0),
                action=action.squeeze(0),
                reward=torch.tensor(reward, dtype=torch.float32),
                value=value.squeeze(0),    
                log_prob=log_prob.squeeze(0),  
                done=done,
                advantage=torch.tensor(0.0),
                ret=torch.tensor(0.0)
            )
            
            steps.append(s)
            state_deuqe_tensor.append(torch.tensor(next_state, dtype=torch.float32).permute(2,0,1) / 255.0)

            if done:
                break
        
        self._calculate_advantages(steps)
        return steps
    
    def _calculate_advantages(self, steps):
        """Calculate advantages using Generalized Advantage Estimation (GAE)"""
        if not steps:
            return
        
        # Get the last value for bootstrapping
        last_value = 0.0 if steps[-1].done else steps[-1].value.item()
        
        advantages = []
        returns = []
        
        # Calculate backwards from the end
        gae = 0.0
        next_value = last_value
        next_return = last_value
        
        for i in reversed(range(len(steps))):
            step = steps[i]
            
            # Calculate TD error for GAE
            delta = step.reward.item() + self.gamma * next_value * (1 - step.done) - step.value.item()
            gae = delta + self.gamma * self.lam * (1 - step.done) * gae
            
            # Fixed: Calculate returns correctly (forward accumulation)
            current_return = step.reward.item() + self.gamma * next_return * (1 - step.done)
            
            # Store for next iteration (going backwards)
            next_value = step.value.item()
            next_return = current_return
            
            advantages.insert(0, gae)
            returns.insert(0, current_return)
    
        # Normalize advantages
        advantages = torch.tensor(advantages, dtype=torch.float32)
        if len(advantages) > 1:  # Avoid division by zero
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Update steps with calculated values
        for i, step in enumerate(steps):
            step.advantage = advantages[i]
            step.ret = torch.tensor(returns[i], dtype=torch.float32)

    def update_policy(self, steps, epochs=4, batch_size=64):
        dataset = PPODataset(steps)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        for epoch in range(epochs):
            for batch in dataloader:
                # Fixed: Evaluate OLD actions under NEW policy (critical for PPO)
                new_log_probs, new_values, entropy = self.net.evaluate_actions(
                    batch['state'], 
                    batch['action']
                )

                # Calculate policy loss with PPO clipping
                ratios = torch.exp(new_log_probs.squeeze() - batch['log_prob'].squeeze())

                surr1 = ratios * batch['advantage']
                surr2 = torch.clamp(ratios, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * batch['advantage']
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss
                value_loss = torch.nn.functional.mse_loss(new_values.squeeze(), batch['ret'])

                # Entropy bonus (we want to maximize entropy, so negative loss)
                entropy_loss = -entropy

                # Total loss
                total_loss = policy_loss + self.value_coef * value_loss + self.entropy_coef * entropy_loss

                # Optimization step
                self.net.optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=1.0)  # Increased from 0.5
                self.net.optimizer.step()

    def train (self, max_episodes = 1000 ,  max_steps_per_episode=300, update_frequency=10):

        episode_rewards = []
        all_steps = []
        for episode in range(max_episodes):
            steps = self.play_episode(max_steps_per_episode)
            all_steps.extend(steps)

            
            total_reward = sum(step.reward.item() for step in steps)
            episode_rewards.append(total_reward)

            if (episode + 1) % update_frequency == 0 or episode == max_episodes - 1:
                print(f"Updating policy at episode {episode + 1}")
                self.update_policy(all_steps)
                all_steps = []

            # Log progress
            if episode % 10 == 0:
                avg_reward = np.mean(episode_rewards[-10:]) if len(episode_rewards) >= 10 else np.mean(episode_rewards)
                print(f"Episode {episode}, Total Reward: {total_reward:.2f}, Avg Reward (last 10): {avg_reward:.2f}")

            # Early stopping if performance is good
            if len(episode_rewards) >= 100 and np.mean(episode_rewards[-100:]) > 500:  # Adjust threshold as needed
                print(f"Training completed! Average reward over last 100 episodes: {np.mean(episode_rewards[-100:]):.2f}")
                break
            
        return episode_rewards
    
    def save_model(self, path):
        """Save the trained model"""
        torch.save(self.net.state_dict(), path)
        print(f"Model saved to {path}")
    
    def load_model(self, path):
        """Load a trained model"""
        self.net.load_state_dict(torch.load(path))
        print(f"Model loaded from {path}")
    

if __name__ == "__main__":

    myPPO = PPO() 
    rewards = myPPO.train()
    myPPO.save_model("ppo_car_racing.pth")





