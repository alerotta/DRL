import gymnasium as gym
import pygame
import torch
import numpy as np
from model import CarRacingCNN
from collections import deque

PATH = "trained_bc_model.pth"

if __name__ == "__main__" :

    env = gym.make("CarRacing-v3" , render_mode = "human")
    state, _  = env.reset()

    pygame.init()
    clock = pygame.time.Clock()

    frame_buffer = deque(maxlen=4)
    processed_frame = state.astype(np.float32) / 255.0  
    for _ in range(4):
        frame_buffer.append(processed_frame)
    

    model = CarRacingCNN(12,3)
    model.load_state_dict(torch.load(PATH, weights_only=True))
    model.eval()

    running = True

    while running :

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        stacked_frames = np.concatenate(frame_buffer, axis=-1)
        stacked_frames = np.transpose(stacked_frames, (2, 0, 1))
        input_tensor = torch.FloatTensor(stacked_frames).unsqueeze(0) 

        with torch.no_grad():
            action_tensor = model(input_tensor)
            action = action_tensor.squeeze(0).numpy()  

        state,_,terminated,truncated,_ = env.step(action)
        processed_frame = state.astype(np.float32) / 255.0
        frame_buffer.append(processed_frame)

        # Reset environment when episode ends
        if terminated or truncated:
            state, _ = env.reset()
            processed_frame = state.astype(np.float32) / 255.0
            frame_buffer.clear()
            for _ in range(4):
                frame_buffer.append(processed_frame)

        clock.tick(30)

    env.close()
    pygame.quit()
