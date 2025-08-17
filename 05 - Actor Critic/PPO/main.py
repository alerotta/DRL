import torch
import torch.nn as nn

from network import MultiHeadNetwork

import gymnasium as gym
import numpy as np

from dataclasses import dataclass

GAMMA = 0.9
LAMBDA = 0.95

@dataclass
class Step :
    obs_t : torch.tensor
    action : torch.tensor
    reward : float
    done : int
    value : torch.tensor
    log_prob : torch.tensor

def collect_rollout (env : gym.Env , net: MultiHeadNetwork, rollout_lenght :  int ):

    obs, _ = env.reset()
    rollout = []
    with torch.no_grad():
        for i in range(rollout_lenght):

            obs_t =  torch.from_numpy(obs)
            logits , value  = net(obs_t)
            dist = torch.distributions.Categorical(logits=logits)
            action_t = dist.sample()
            log_prob = dist.log_prob(action_t)

            action = int(action_t.item())

            next_obs , reward , is_done, is_trunc , _  = env.step(action)
            
            done = 0
            if (is_done or is_trunc) : done = 1

            step = Step(obs_t,action_t,float(reward),done,value,log_prob)
            rollout.append(step)

            obs = next_obs

            if done and i == rollout_lenght -1 :
                last_val = 0

            if  not done and i == rollout_lenght -1 :
                obs_t =  torch.from_numpy(obs)
                _ , last_val = net(obs_t)

            if done :
                obs, _ = env.reset()
        
    return rollout , last_val
    

def calculate_advantages (rollout, last_value) :
   
    advantages = []
    

    last_gae = 0
    for i in reversed(range(len(rollout))) : 
        if rollout[i].done:
            next_value = 0
        else:
            next_value = rollout[i+1].value.item() if i < len(rollout)-1 else last_value.item()

        delta = rollout[i].reward + GAMMA * next_value - rollout[i].value.item()
        last_gae = delta + GAMMA * LAMBDA * (1 - rollout[i].done) * last_gae
        advantages.append(last_gae)

    advantages.reverse()    
    return advantages


        
def train(env : gym.Env , net: MultiHeadNetwork, rollout_lenght :  int, epochs=4, batch_size=64, clip_eps=0.2, lr=3e-4) :

    optimizer = torch.optim.Adam(net.parameters(), lr=lr)

    for episode in range(1000):  # Number of training episodes
        # 1. Collect rollout
        rollout, last_value = collect_rollout(env, net, rollout_lenght)
        advantages = calculate_advantages(rollout, last_value)
        returns = [step.value.item() + adv for step, adv in zip(rollout, advantages)]

        # 2. Flatten rollout data
        obs_batch = torch.stack([step.obs_t.float() for step in rollout])
        action_batch = torch.stack([step.action for step in rollout])
        old_logprobs = torch.stack([step.log_prob for step in rollout])
        returns = torch.tensor(returns, dtype=torch.float32)
        advantages = torch.tensor(advantages, dtype=torch.float32)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)  # Normalize

        # 3. PPO update
        for _ in range(epochs):
            idxs = np.arange(len(rollout))
            np.random.shuffle(idxs)
            for start in range(0, len(rollout), batch_size):
                end = start + batch_size
                mb_idx = idxs[start:end]

                mb_obs = obs_batch[mb_idx]
                mb_actions = action_batch[mb_idx]
                mb_old_logprobs = old_logprobs[mb_idx]
                mb_returns = returns[mb_idx]
                mb_advantages = advantages[mb_idx]

                logits, values = net(mb_obs)
                dist = torch.distributions.Categorical(logits=logits)
                new_logprobs = dist.log_prob(mb_actions)

                ratio = (new_logprobs - mb_old_logprobs).exp()
                surr1 = ratio * mb_advantages
                surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * mb_advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                value_loss = (mb_returns - values.squeeze(-1)).pow(2).mean()

                loss = policy_loss + 0.5 * value_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        # Optional: print progress
        if episode % 10 == 0:
            print(f"Episode {episode}, loss: {loss.item():.4f}")


if __name__ == "__main__" :
    
    env = gym.make("LunarLander-v3")
    net = MultiHeadNetwork(env.observation_space.shape[0], env.action_space.n)

    rollout_length = 2048
    train(env, net, rollout_length)

    env.close()
 