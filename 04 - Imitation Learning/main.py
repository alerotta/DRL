import gymnasium as gym 
import pygame 
import numpy as np 
import os 
from datetime import datetime 
from collections import deque

def keys_to_action ():

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

    save_dir = "trajectories"
    os.makedirs(save_dir, exist_ok=True)

    #placed here it means that to create more than one trajectory the application must be stopped!
    trajectory = []
    frame_buffer = deque(maxlen=4)

    env = gym.make("CarRacing-v3", render_mode='human', lap_complete_percent=1.0 )
    obsv, info = env.reset()

    pygame.init()
    clock = pygame.time.Clock()

    running = True
    total_reward = 0.0 
    max_total_reward = 500

    while running :

        

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

        if total_reward > max_total_reward : 
        
            timestamp = datetime.now().strftime("%m%d_%H%M%S")
            filename = os.path.join(save_dir, f"traj_{timestamp}.npz")
            states = np.array([np.stack(step[0]) for step in trajectory], dtype=np.uint8)
            actions = np.array([step[1] for step in trajectory], dtype=np.float32)
            np.savez_compressed(filename, states=states, actions=actions)
            print(f"Saved trajectory with {len(states)} steps to {filename}")

            obsv, info = env.reset()
            frame_buffer.clear()
            trajectory = []
            total_reward = 0.0

        
        
        action = keys_to_action()
        obsv , reward , is_done , is_trunc , _ = env.step(action)
        frame_buffer.append(obsv)
        if len(frame_buffer) == 4:
            trajectory.append((list(frame_buffer), action.copy()))

        total_reward += float(reward)


        


        clock.tick(30)

    env.close()
    pygame.quit()



   # print(total_reward)
   # timestamp = datetime.now().strftime("%m%d_%H%M%S")
   # filename = os.path.join(save_dir, f"traj_{timestamp}.npz")
   # states = np.array([np.stack(step[0]) for step in trajectory], dtype=np.uint8)
   # actions = np.array([step[1] for step in trajectory], dtype=np.float32)
   # np.savez_compressed(filename, states=states, actions=actions)
   # print(f"Saved trajectory with {len(states)} steps to {filename}")


