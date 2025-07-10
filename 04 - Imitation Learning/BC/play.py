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
    frame_count = 0

    frame_buffer = deque(maxlen=4)

    model = CarRacingCNN(12,3)
    model.load_state_dict(torch.load(PATH, weights_only=True))
    model.eval()

    running = True

    while running :

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
        
        state = state / 255.0
        state = torch.FloatTensor(state)
        frame_buffer.append(state)

        if  frame_count > 40:
            with torch.no_grad():
                input_tensor = torch.stack(list(frame_buffer)).unsqueeze(0)
                action = model(input_tensor)
                action = action.detach().numpy().squeeze() 
                print (action)
                state, _ , _ ,_ ,_ = env.step(action)
            
        frame_count += 1
        clock.tick(30)

    env.close()
    pygame.quit()
