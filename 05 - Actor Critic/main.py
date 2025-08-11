import torch
import torch.nn as nn
import torch.optim as optim

import gymnasium as gym 
import numpy as np

from network import MultiHeadNetwork
from dataclasses import dataclass

import typing as tt

@dataclass
class Step : 
    obs : np.array
    action : int 
    reward : float
    done : bool
    value : float
    log_prob : float  


@dataclass 
class Rollout: 
    steps : tt.List[Step]


def collect_rollout ( env: gym.Env , net: MultiHeadNetwork , rollout_length: int ):
    obs ,  _ = env.reset()
    steps = []
    net.eval()
    with torch.no_grad():
        for  _ in range(rollout_length):
            # Create tensor correctly
            obs_tensor = torch.as_tensor(obs, dtype=torch.float32)

            action_logits , value = net(obs_tensor)
            dist = torch.distributions.Categorical(logits=action_logits)
            action_t = dist.sample()
            log_prob_t = dist.log_prob(action_t)

            action = int(action_t.item())
            next_obs , reward, is_done, is_trunc , _ = env.step(action)
            done = bool(is_done or is_trunc)

            # Store numpy obs and scalar values to make stacking easy later
            steps.append(
                Step(
                    obs=np.array(obs, copy=True),
                    action=action,
                    reward=float(reward),
                    done=done,
                    value=float(value.item()),
                    log_prob=float(log_prob_t.item()),
                )
            )

            obs = next_obs
            if done : 
                obs ,  _ = env.reset()

    last_value = float(net(torch.as_tensor(obs, dtype=torch.float32))[1].item())
    return Rollout(steps=steps), obs, last_value


def train (env: gym.Env ,
            net: MultiHeadNetwork, 
            optimizer : optim.Optimizer,
            rollout_length : int = 128, 
            gamma: float = 0.99,
            entropy_coef: float = 0.01,
            value_coef: float = 0.5,
            max_grad_norm: float = 0.5,
            updates: int = 1000) : 
    
    for update in range (updates):
        rollout , _ ,last_value = collect_rollout(env,net,rollout_length)

        # Stack rollout
        obs_batch = torch.as_tensor(np.stack([s.obs for s in rollout.steps]), dtype=torch.float32)
        actions = torch.as_tensor([s.action for s in rollout.steps], dtype=torch.int64)
        rewards = torch.as_tensor([s.reward for s in rollout.steps], dtype=torch.float32)
        dones = torch.as_tensor([s.done for s in rollout.steps], dtype=torch.float32)

        returns = torch.empty(rollout_length, dtype=torch.float32)
        next_ret = torch.as_tensor(last_value, dtype=torch.float32)

        for t in reversed(range(rollout_length)):
            next_ret = rewards[t] + gamma * next_ret * (1.0 - dones[t])
            returns[t] = next_ret

        net.train()
        logits, values = net(obs_batch)              # logits: [B, A], values: [B]
        dist = torch.distributions.Categorical(logits=logits)
        log_probs = dist.log_prob(actions)           # [B]
        entropy = dist.entropy().mean()

        advantages = returns - values.detach()
        advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)

        policy_loss = -(advantages * log_probs).mean()
        value_loss = 0.5 * (returns - values).pow(2).mean()
        loss = policy_loss + value_coef * value_loss - entropy_coef * entropy

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), max_grad_norm)
        optimizer.step()

        if (update + 1) % 10 == 0:
            print(f"update {update+1}: loss={loss.item():.3f} "
                  f"policy={policy_loss.item():.3f} value={value_loss.item():.3f} "
                  f"entropy={entropy.item():.3f} avg_r={rewards.mean().item():.2f}")
            
if __name__ == "__main__":
    env = gym.make("CartPole-v1")
    obs_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n
    net = MultiHeadNetwork(obs_dim, n_actions)
    optimizer = torch.optim.Adam(net.parameters(), lr=3e-4)
    train(env, net, optimizer, rollout_length=128, updates=500)



    

        







