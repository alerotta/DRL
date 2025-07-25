# GAIL Implementation (CarRacing-v3)

This directory contains an implementation of **Generative Adversarial Imitation Learning (GAIL)** applied to the `CarRacing-v3` environment from Gymnasium.

## 📁 Contents

- `expert.py`  
  Manually collect expert trajectories by playing the game. Saves data in a format usable by GAIL.

- `discriminator.py`  
  Defines the neural network that distinguishes expert state-action pairs from those generated by the agent.

- `train.py`  
  Trains both the discriminator and a PPO agent (via `stable-baselines3`) in a GAIL loop.

- `play.py`  
  Runs the trained policy autonomously to visualize agent performance.

- `test.py`  
  Verifies that expert trajectory data is correctly saved and formatted.

---

## ⚠️ Notes

- **Training GAIL is slow**, especially for pixel-based environments like CarRacing. Expect to train for hundreds of iterations before meaningful results appear.
- This implementation is a **starting point** and may require further tuning or debugging for optimal results (e.g., reward normalization, discriminator regularization, replay buffer).
