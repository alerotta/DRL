import torch
import torch.nn as nn
import torch.optim as optim


import gymnasium as gym 
import numpy as np

from network import MultiHeadNetwork
from dataclasses import dataclass

import typing as tt

GAMMA = 0.99

@dataclass
class Step : 
    obs : torch.Tensor
    action : int 
    reward : float
    done : bool
    value : torch.Tensor
    log_prob : torch.Tensor 


def collect_rollout ( env : gym.Env , net : MultiHeadNetwork , rollout_size : int ):

    rollout = []
    obs , _ = env.reset()
    
    with torch.no_grad():
        for _ in range(rollout_size):

            obs_tensor = torch.as_tensor(obs, dtype=torch.float32)
            logits, value = net(obs_tensor)
            dist = torch.distributions.Categorical(logits=logits)
            action_t = dist.sample()
            log_prob = dist.log_prob(action_t)

            action = int(action_t.item())
            next_obs, reward , is_done , is_trunc , _ = env.step(action)
            done = bool(is_done or is_trunc)
            step = Step(obs_tensor,action,float(reward),done,value,log_prob)
            rollout.append(step)
            obs = next_obs
            if done : 
                obs , _ = env.reset()
    
        last_step = rollout[-1]
        if last_step.done : 
            last_value = torch.tensor(0.0)
        else :
            obs_tensor = torch.as_tensor(obs, dtype=torch.float32)
            _, last_value = net(obs_tensor)
 
    return rollout , last_value




def calculate_targets (rollout : tt.List[Step], last_value : torch.Tensor ) : 

    targets = []

    for i, step in enumerate(rollout):
        if step.done:
            target = torch.as_tensor(step.reward, dtype=torch.float32)
        elif i == len(rollout) - 1:
            target = torch.as_tensor(step.reward, dtype=torch.float32) + GAMMA * last_value
        else:
            target = torch.as_tensor(step.reward, dtype=torch.float32) + GAMMA * rollout[i+1].value
        # Ensure target is always shape [1]
        target = target.reshape(1)
        targets.append(target)
    return torch.stack(targets)

def calculate_advantages (rollout : tt.List[Step] , targets :  torch.Tensor):

    adv = torch.zeros_like(targets)
    for i,step in enumerate(rollout):
        adv[i] = targets[i] - torch.as_tensor(step.value, dtype=torch.float32)
    return adv


def train (env : gym.Env,
        net : MultiHeadNetwork,
        optimizer : torch.optim.Optimizer,
        rollout_size : int,
        n_updates : int,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01
):
    for update in range(n_updates):
        
        rollout , last_val = collect_rollout(env,net,rollout_size)
        targets = calculate_targets(rollout,last_val)
        adv = calculate_advantages(rollout,targets)

        obs_batch = torch.stack([step.obs for step in rollout])
        action_batch = torch.tensor([step.action for step in rollout])
        log_prob_batch = torch.stack([step.log_prob for step in rollout])
        value_batch = torch.stack([step.value for step in rollout]).squeeze(-1)

        logits, values = net(obs_batch)
        dist = torch.distributions.Categorical(logits=logits)
        new_log_probs = dist.log_prob(action_batch)
        entropy = dist.entropy().mean()

        policy_loss = -(adv.detach().squeeze() * new_log_probs).mean()
        value_loss = value_coef * (targets.squeeze() - values.squeeze()).pow(2).mean()
        entropy_loss = -entropy_coef * entropy
        loss = policy_loss + value_loss + entropy_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (update + 1) % 10 == 0:
            print(f"Update {update+1}/{n_updates}, Loss: {loss.item():.3f}, Policy Loss: {policy_loss.item():.3f}, Value Loss: {value_loss.item():.3f}, Entropy: {entropy.item():.3f}")




if __name__ == "__main__" : 

    env = gym.make("LunarLander-v3")
    net = MultiHeadNetwork(8,4)
    optimizer = optim.Adam(net.parameters(), lr=1e-3)
    train(env, net, optimizer, rollout_size=200, n_updates=1000)



 




            

            

