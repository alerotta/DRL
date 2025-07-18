import numpy as np
import os 
import glob
import matplotlib.pyplot as plt

if __name__ == "__main__":
    trajectory_dir = "trajectories"
    files = glob.glob(os.path.join(trajectory_dir, "*npz"))

    file = files[0]
    data = np.load(file)
    states = data['states']  
    actions = data['actions']  
    next_states = data['next_states']

    print (f"there are {len(states)} states and {len(actions)} actions")
    print(f"Shape of a states entry: {states[0].shape}")
    
    # Visualize all four frames from states[0] and next_states[0] together
    # states[0] and next_states[0] have shape (4, 96, 96, 3) - we'll plot all 8 frames
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    
    # Top row: states[0]
    for i in range(4):
        frame = states[200][i]  # Shape: (96, 96, 3)
        axes[0, i].imshow(frame.astype(np.uint8))
        axes[0, i].set_title(f"State Frame {i+1}")
        axes[0, i].axis('off')
    
    # Bottom row: next_states[0]
    for i in range(4):
        frame = next_states[200][i]  # Shape: (96, 96, 3)
        axes[1, i].imshow(frame.astype(np.uint8))
        axes[1, i].set_title(f"Next State Frame {i+1}")
        axes[1, i].axis('off')
    
    plt.tight_layout()
    plt.show()

