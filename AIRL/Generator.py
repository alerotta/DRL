import Network as net
import gymnasium as gym
 
import torch
import torch.nn.functional as F
from torch.utils.data import  DataLoader 
from collections import deque 

from Data import Step ,AIRLDataset
from dataclasses import replace

class PPO :

    def __init__(self , env : gym.Env, 
                discriminator = None,
                state_encoder = None,
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

        self.env = env 
        self.state_encoder = state_encoder  or net.CNNEncoder(self.env.observation_space.shape)
        self.network = net.PPONetwork(self.env.observation_space.shape , self.env.action_space.shape[0] , hidden_size , shared_encoder=self.state_encoder)
        self.discriminator = discriminator

    
    def play_episode (self, maxlen):

        steps = []
        frame, _ = self.env.reset()
        state_deque_tensor = deque(maxlen=4) 

        for _ in range(4):
            #permute to enter conv 2d
            state_deque_tensor.append(torch.tensor(frame , dtype=torch.float32).permute(2,0,1) / 255.0)

        #unsqueeze for conv2d
        tensor = torch.cat(list(state_deque_tensor),dim = 0).unsqueeze(0)

        for i in range(maxlen):

            with torch.no_grad():
                action , log_prob , value = self.network.sample_action(tensor)
                
            next_frame , _ , is_terminated , is_truncated , _ = self.env.step(action.squeeze(0).numpy())
            state_deque_tensor.append(torch.tensor(next_frame , dtype=torch.float32).permute(2,0,1)/ 255.0)
            tensor_next = torch.cat(list(state_deque_tensor),dim = 0).unsqueeze(0)
            reward = self.discriminator.reward(tensor,action,tensor_next).squeeze(0)
            done = is_terminated or is_truncated 

            s =Step(
                state=tensor.squeeze(0),
                next_state=tensor_next.squeeze(0),
                action=action.squeeze(0),
                log_prob=log_prob.squeeze(0),
                value=value.squeeze(0),
                reward= reward,
                done=done,
                advantage= torch.tensor(0.0),
                ret = torch.tensor(0.0),
                expert= False
            )

            tensor = tensor_next
            if done and i < maxlen -1 :
                steps.append(s)
                frame, _ = self.env.reset()
                state_deque_tensor.clear() 
                for _ in range(4):
                    #permute to enter conv 2d
                    state_deque_tensor.append(torch.tensor(frame , dtype=torch.float32).permute(2,0,1) / 255.0)
                tensor = torch.cat(list(state_deque_tensor),dim = 0).unsqueeze(0)
                continue
            
            if done and i == maxlen -1 :
                s.next_state =None
                steps.append(s)
                break
            

            steps.append(s)
        if done :
            self._calculate_advantages(steps, last_next_value = 0)
        else :
            _ , _ , value = self.network.sample_action(tensor)
            self._calculate_advantages(steps,value.squeeze(0))
        return steps
    
    def _calculate_advantages(self, steps, last_next_value = None):

        gamma , lam = self.gamma , self.lam

        if last_next_value is None :
            last_next_value = torch.zeros_like(steps[-1].value)
        
        with torch.no_grad():
            gae = torch.zeros_like(steps[-1].value)

            for t in reversed(range(len(steps))):
                v_t = steps[t].value  
                r_t = steps[t].reward 
                done_t = torch.tensor(float(steps[t].done))
                not_done = 1.0 - done_t

                v_tp1 = steps[t+1].value if (t + 1 < len(steps)) else last_next_value
                delta = r_t + gamma * not_done * v_tp1 - v_t
                gae = delta + gamma * lam * not_done * gae
                ret_t = gae + v_t
                steps[t] = replace(steps[t], advantage=gae.detach(), ret=ret_t.detach())
        
        advs = torch.stack([s.advantage for s in steps])
        advs = (advs - advs.mean()) / (advs.std(unbiased=False) + 1e-8)
        for t in range(len(steps)):
            steps[t] = replace(steps[t], advantage=advs[t])
        
    def update_policy ( self, steps, epochs = 4 , batch_size = 64):

        dataset = AIRLDataset(steps=steps)
        dataloader = DataLoader(dataset , batch_size=batch_size , shuffle=True)

        for epoch in range(epochs):
            for batch in dataloader:

                states = batch['state']
                actions = batch['action']
                old_log_probs = batch['log_prob'].squeeze(-1) 
                returns = batch['ret'].squeeze(-1)           
                advantages = batch['advantage'].squeeze(-1)

                new_log_probs, values, entropy = self.network.evaluate_actions(states, actions)
                new_log_probs = new_log_probs.squeeze(-1)              
                values = values.squeeze(-1)  
                ratio = torch.exp(new_log_probs - old_log_probs)
                    
                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                value_loss = 0.5 * (returns - values).pow(2).mean()
                loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy

                self.network.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), 0.5)
                self.network.optimizer.step()







        
            
    




