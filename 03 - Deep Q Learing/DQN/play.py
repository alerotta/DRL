import gymnasium as gym 
import pygame 
import ale_py

def key_to_action () : 
    
    action = 0
    keys = pygame.key.get_pressed()

    
    if keys[pygame.K_RIGHT] or keys[pygame.K_d]:
        action = 2
    elif keys[pygame.K_LEFT] or keys[pygame.K_a]:
        action = 3
    elif keys[pygame.K_SPACE] :
        action = 1

    return action 

if __name__ == "__main__" : 

    gym.register_envs(ale_py)
    env = gym.make("BreakoutNoFrameskip-v4", render_mode = "human")
    obs, info = env.reset()

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
                if event.key == pygame.K_r:
                    obsv, info = env.reset()
        
        action = key_to_action()
        obsv , _ , _ , _ , _ = env.step(action)

        clock.tick(30)
    
    env.close()
    pygame.quit()
