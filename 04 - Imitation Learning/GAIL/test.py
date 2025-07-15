import numpy as np
import os 
import glob 
import matplotlib.pyplot as plt

DIR = "trajectories"

if __name__ == "__main__" :
    
    
    trajectory_file = glob.glob(os.path.join(DIR, "*.npz"))
    print(trajectory_file)
    data = np.load(trajectory_file[0])
    states = data['states']  
    actions = data['actions']  

    print (f"there are {len(states)} states and {len(actions)} actions")
    print(f"Shape of a states entry: {states[0].shape}")
        
    first_frame = states[0][0] 

    plt.figure(figsize=(8, 8))
    plt.imshow(first_frame.astype(np.uint8))
    plt.title("First frame of the first state")
    plt.axis('off')
    plt.show()


    
    

    