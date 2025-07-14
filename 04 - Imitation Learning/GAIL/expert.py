import gymnasium as gym 
import pygame 
import numpy as np 
import os 
from datetime import datetime 
from collections import deque



def keys_to_action ():

    """ 
    used to map keyboard input into env inputs,
    the envirmonment reuqests a numpy array of three floats
    fist element [-1,1] where -1 means left and 1 means right
    second element [0,1] used to controll the gas
    third element [0,1] used to controll the brake"""

    keys = pygame.key.get_pressed()
    action = np.array([0.0,0.0,0.0])

    if keys[pygame.K_RIGHT] or keys[pygame.K_d]:
        action[0] = 1.0
    elif keys[pygame.K_LEFT] or keys[pygame.K_a]:
        action[0] = -1.0
    
    if keys[pygame.K_UP] or keys[pygame.K_w]:
        action[1] = 1.0

    if keys[pygame.K_DOWN] or keys[pygame.K_s]:
        action[2] = 1.0

    return action

if __name__ == '__main__' : 

    # specify/create directory used to store expert demostations
    save_dir = "trajectories"
    os.makedirs(save_dir, exist_ok=True)

    
    trajectory = []

    # environment init
    env = gym.make("CarRacing-v3", render_mode='human', lap_complete_percent=1.0 )
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
                    
                    trajectory = []
                    total_reward = 0.0
                    frame_count = 0


        if total_reward > max_total_reward : 
        
            timestamp = datetime.now().strftime("%m%d_%H%M%S")
            filename = os.path.join(save_dir, f"traj_{timestamp}.npz")
            states = np.array([np.stack(step[0]) for step in trajectory], dtype=np.float32)
            actions = np.array([step[1] for step in trajectory], dtype=np.float32)
            np.savez_compressed(filename, states=states, actions=actions)
            print(f"Saved trajectory with {len(states)} steps to {filename}")

            obsv, info = env.reset()
            trajectory = []
            total_reward = 0.0
            frame_count = 0

        action = keys_to_action()
        #start to collect the trajectory after the first 40 frames, they do not contain useful info.
        if  frame_count > 40:
            trajectory.append([obsv, action])

        obsv , reward , is_done , is_trunc , _ = env.step(action)
        frame_count += 1

      
        

        total_reward += float(reward)


        clock.tick(30)

    env.close()
    pygame.quit()