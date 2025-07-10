# Behavioral Cloning Implementation

This directory contains an implementation of Behavioral Cloning (BC) for the CarRacing-v3 environment from OpenAI Gym.

## 📁 Contents

- `expert.py`  
  Allows you to manually play the CarRacing-v3 game and collect expert trajectories. These are used later for training.

- `model.py`  
  Defines the neural network used for policy learning in the BC setup.

- `train.py`  
  Trains the neural network using the expert trajectories collected by `expert.py`. The trained model is saved automatically.

- `play.py`  
  Runs the trained model to play the game autonomously using the saved weights.

- `test.py`  
  Contains scripts to verify that expert trajectory data is saved correctly and in the right format.

- **Trained BC Model**  
  A pre-trained Behavioral Cloning model is included for immediate use with `play.py`.

---

## 🚀 How to Use

### ✅ Option 1: Use the Pretrained Model

1. Make sure all necessary dependencies are installed (e.g., Gym, PyTorch, etc.).
2. Run the following command to watch the pretrained agent play:

python play.py

### ✅ Option 2: generate trajectories, train the model and use it.

1. Make sure all necessary dependencies are installed (e.g., Gym, PyTorch, etc.).
2. delete pre trained model file.
3. Run the following command to collect trajectories:

python expert.py

4. Run the following command to train the model:

python train.py

5. Run the following command to watch the pretrained agent play:

python play.py
