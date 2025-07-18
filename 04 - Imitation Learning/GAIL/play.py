import gymnasium as gym
import pygame
import torch
import numpy as np
from stable_baselines3 import PPO

MODEL_PATH = "gail_ppo_model"

if __name__ == "__main__":
    
    # Load the trained GAIL PPO model
    print(f"Loading model from {MODEL_PATH}...")
    model = PPO.load(MODEL_PATH)
    print("Model loaded successfully!")
    
    # Create environment with human rendering
    env = gym.make("CarRacing-v3", render_mode="human")
    obs, _ = env.reset()
    
    pygame.init()
    clock = pygame.time.Clock()
    running = True

    
    while running: 


        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False

                
            
        action, _ = model.predict(obs)
        obs, reward, done, truncated, info = env.step(action)
            
            
           
        clock.tick(30)  # 30 FPS
        

    
env.close()
pygame.quit()
print("Game finished!")
