import os
import pygame 
import gymnasium as gym 
import numpy as np
from collections import deque
from datetime import datetime 

def key_to_action():

    keys = pygame.key.get_pressed()
    action = np.array([0.0,0.0,0.0])

    if keys[pygame.K_d] or keys[pygame.K_RIGHT]: 
        action[0] = 1.0
    elif keys[pygame.K_a] or keys[pygame.K_LEFT]:
        action[0] = -1.0

    if keys[pygame.K_w] or keys[pygame.K_UP]: 
        action[1] = 1.0

    if keys[pygame.K_s] or keys[pygame.K_DOWN]: 
        action[2] = 1.0
    
    return action


if __name__ == "__main__": 

    # specify/create directory used to store expert demostations
    save_dir = "trajectories"
    os.makedirs(save_dir, exist_ok=True)

    trajectory = []
    frame_buffer = deque(maxlen=4)
    frame_buffer_next = deque(maxlen=4)

    env = gym.make("CarRacing-v3", render_mode= "human")
    obsv, info = env.reset()

    pygame.init()
    clock = pygame.time.Clock()

    running = True
    total_reward = 0.0 
    max_total_reward = 500
    frame_count = 0

    while running :

        
        # check for quit/reset the game
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                if event.key == pygame.K_r:

                    obsv, info = env.reset()
                    frame_buffer.clear()
                    trajectory = []
                    total_reward = 0.0
                    frame_count = 0


        if total_reward > max_total_reward : 
        
            timestamp = datetime.now().strftime("%m%d_%H%M%S")
            filename = os.path.join(save_dir, f"traj_{timestamp}.npz")
            states = np.array([np.stack(step[0]) for step in trajectory], dtype=np.float32)
            actions = np.array([step[1] for step in trajectory], dtype=np.float32)
            next_states = np.array([np.stack(step[2]) for step in trajectory], dtype=np.float32)
            np.savez_compressed(filename, states=states, actions=actions, next_states=next_states)
            print(f"Saved trajectory with {len(states)} steps to {filename}")

            obsv, info = env.reset()
            frame_buffer.clear()
            trajectory = []
            total_reward = 0.0
            frame_count = 0

        
        
        action = key_to_action()
        frame_buffer.append(obsv)
        frame_buffer_next = frame_buffer.copy()
        obsv , reward , is_done , is_trunc , _ = env.step(action)
        frame_buffer_next.append(obsv)
        frame_count += 1

      

        #start to collect the trajectory after the first 40 frames, they do not contain useful info.
        if  frame_count > 40:
            trajectory.append([list(frame_buffer), action , list(frame_buffer_next)])

      
        

        total_reward += float(reward)


        


        clock.tick(30)

    env.close()
    pygame.quit()