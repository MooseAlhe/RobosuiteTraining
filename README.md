# 🧠 TD3 Robosuite Agent

This repository implements a **Twin Delayed Deep Deterministic Policy Gradient (TD3)** reinforcement learning agent using **PyTorch** and **Robosuite**.  
The agent controls a **Panda robotic arm** to perform continuous control tasks (e.g., opening a door) in a simulated environment built with Robosuite’s Gym API.

---

## 📁 Project Structure

```
.
├── buffer.py               # Replay buffer implementation for storing experiences
├── experiment_runner.py    # Hyperparameter tuning utility for multiple experiment runs
├── main.py                 # Core training entrypoint with TensorBoard logging
├── test.py                 # Evaluation and visualization of trained agent
├── networks.py             # Actor and Critic neural network definitions
├── td3_torch.py            # Core TD3 algorithm and agent implementation
├── requirements.txt        # Python dependencies
├── .gitignore
```

---

## ⚙️ Requirements and Setup

### 1. Environment

Recommended setup:
- **Python:** 3.11  
- **OS:** Windows 10/11, Ubuntu, or macOS  
- **GPU (optional):** CUDA-enabled NVIDIA GPU for faster training  

### 2. Installation

Create and activate a conda environment:

```bash
conda create -n td3_robosuite python=3.11
conda activate td3_robosuite
```

Then install dependencies:

```bash
pip install -r requirements.txt
```

**Key dependencies** (`requirements.txt`):

```
gym==0.23.0
pybullet==3.2.6
matplotlib==3.8.2
tensorboard==2.15.1
robosuite==1.4.0
termcolor==2.4.0
h5py==3.10.0
torch
torchvision
torchaudio
```

---

## 🧩 Core Components

### `buffer.py` — Replay Buffer
Stores transitions `(state, action, reward, next_state, done)` and supports random sampling for training batches.  
Implements circular memory overwrite when full.

### `networks.py` — Actor & Critic Networks
- **Actor:** Maps environment states to continuous actions (bounded via `tanh`).
- **Critic:** Estimates Q-values for `(state, action)` pairs using two independent networks to reduce overestimation bias.

### `td3_torch.py` — TD3 Agent
Implements the **Twin Delayed Deep Deterministic Policy Gradient** algorithm:
- Double critics for bias reduction  
- Target networks for stable updates  
- Soft update mechanism (`τ`)  
- Gaussian exploration noise  
- Replay buffer–based experience replay  

### `main.py` — Training Script
Trains the TD3 agent on the **Panda Door** environment.  
Includes TensorBoard logging and automatic checkpoint saving every 10 episodes.

### `experiment_runner.py` — Hyperparameter Optimization
Runs multiple experiments automatically using grid search across key hyperparameters:
- Learning rates (`alpha`, `beta`)
- Exploration noise
- Batch size

### `test.py` — Evaluation
Loads pre-trained models and visualizes learned policies in a rendered environment, showing the robot interacting with the task.

---

## 🧠 Training Instructions

### 1. Train the Agent

```bash
python main.py
```

Training progress and saved model checkpoints will appear under:

```
logs/
tmp/td3/
```

### 2. Monitor Training in TensorBoard

```bash
tensorboard --logdir logs
```

This opens a local server (default: [http://localhost:6006](http://localhost:6006)) to visualize learning curves and rewards.

### 3. Run Hyperparameter Experiments

Run multiple training sessions automatically:

```bash
python experiment_runner.py
```

Each experiment is logged in:

```
logs/experiment_<ID>/
```

### 4. Test the Trained Model

Render the robot executing the learned behavior:

```bash
python test.py
```

You’ll see a real-time simulation of the **Panda robot** interacting with the environment (e.g., opening the door).

---

## 📊 Logging and Checkpointing

- Models (`actor`, `critic_1`, `critic_2`, and target networks) are automatically saved every 10 episodes:

  ```
  tmp/td3/
  ```

- TensorBoard logs each episode’s score and experiment identifier for easy comparison across training runs.

---

## ⚙️ Key Hyperparameters

| Parameter | Description | Default |
|------------|--------------|----------|
| `alpha` | Actor learning rate | 0.001 |
| `beta` | Critic learning rate | 0.001 |
| `gamma` | Discount factor for future rewards | 0.99 |
| `tau` | Target network soft update rate | 0.005 |
| `batch_size` | Training batch size | 128 |
| `noise` | Action exploration noise | 0.1 |
| `update_actor_interval` | Frequency of actor updates | 2 |
| `warmup` | Steps before using the policy (random actions) | 1000 |

---

## 🧪 Example Output

**Training Output:**

```
Episode: 50  Score: 124.7
Models saved
Episode: 51  Score: 137.2
```

**Testing Output:**

```
Episode: 0  Score: 142.9
Episode: 1  Score: 150.3
```