"""
Enhanced Dueling Deep Q-Network (DQN) Agent
Multi-Route Multi-Class Airline Revenue Management
File: agents/model.py
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
import os


class DuelingDQNNetwork(nn.Module):
    """
    Dueling DQN Architecture.
    Separates state-value (V) and advantage (A) estimation, which is highly
    effective in pricing environments where many actions yield similar results.
    """
    
    def __init__(self, state_size, action_size, hidden_size=256):
        super(DuelingDQNNetwork, self).__init__()
        
        # Shared feature representation
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.ln1 = nn.LayerNorm(hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.ln2 = nn.LayerNorm(hidden_size)
        
        # Value Stream (V)
        self.value_fc1 = nn.Linear(hidden_size, hidden_size // 2)
        self.value_ln1 = nn.LayerNorm(hidden_size // 2)
        self.value_fc2 = nn.Linear(hidden_size // 2, 1)
        
        # Advantage Stream (A)
        self.adv_fc1 = nn.Linear(hidden_size, hidden_size // 2)
        self.adv_ln1 = nn.LayerNorm(hidden_size // 2)
        self.adv_fc2 = nn.Linear(hidden_size // 2, action_size)
        
        self.leaky_relu = nn.LeakyReLU(0.01)
        self.dropout = nn.Dropout(0.2)
        
        self._initialize_weights()
        
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        
    def forward(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(0)
            
        # Shared features
        x = self.leaky_relu(self.ln1(self.fc1(x)))
        x = self.dropout(x)
        x = self.leaky_relu(self.ln2(self.fc2(x)))
        x = self.dropout(x)
        
        # Value stream
        v = self.leaky_relu(self.value_ln1(self.value_fc1(x)))
        v = self.value_fc2(v)
        
        # Advantage stream
        a = self.leaky_relu(self.adv_ln1(self.adv_fc1(x)))
        a = self.adv_fc2(a)
        
        # Recombine: Q(s,a) = V(s) + (A(s,a) - mean(A(s,a)))
        q = v + (a - a.mean(dim=1, keepdim=True))
        return q


class StaticPrioritizedReplayBuffer:
    """
    High-Performance Static NumPy Prioritized Replay Buffer.
    Eliminates dynamic list allocations and frequent `np.stack` operations.
    Returns n-step experiences seamlessly.
    """
    
    def __init__(self, state_size, capacity=50000, alpha=0.6, beta=0.4, beta_increment=0.001):
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        
        # Pre-allocate static NumPy arrays
        self.states      = np.zeros((capacity, state_size), dtype=np.float32)
        self.actions     = np.zeros(capacity, dtype=np.int32)
        self.rewards     = np.zeros(capacity, dtype=np.float32)
        self.next_states = np.zeros((capacity, state_size), dtype=np.float32)
        self.dones       = np.zeros(capacity, dtype=np.float32)
        
        # SumTree approximations for priorities (flattened array for speed)
        self.priorities  = np.zeros(capacity, dtype=np.float32)
        
        self.position = 0
        self.size = 0
        self.max_priority = 1.0
        
    def push(self, state, action, reward, next_state, done):
        """O(1) insertion into pre-allocated buffer."""
        idx = self.position
        
        self.states[idx]      = state
        self.actions[idx]     = action
        self.rewards[idx]     = reward
        self.next_states[idx] = next_state
        self.dones[idx]       = float(done)
        
        self.priorities[idx]  = self.max_priority
        
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
    
    def sample(self, batch_size):
        if self.size < batch_size:
            return None
            
        # Calculate probabilities
        priorities = self.priorities[:self.size]
        probabilities = priorities ** self.alpha
        probabilities /= probabilities.sum()
        
        # Sample indices
        indices = np.random.choice(self.size, batch_size, p=probabilities, replace=False)
        
        # Calculate importance sampling weights
        self.beta = min(1.0, self.beta + self.beta_increment)
        weights = (self.size * probabilities[indices]) ** (-self.beta)
        weights /= weights.max()  # Normalize for stability
        
        return (
            self.states[indices],
            self.actions[indices],
            self.rewards[indices],
            self.next_states[indices],
            self.dones[indices],
            indices,
            weights
        )
        
    def update_priorities(self, indices, priorities):
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority + 1e-5
            self.max_priority = max(self.max_priority, priority)
            
    def __len__(self):
        return self.size


class NStepReplayBuffer(StaticPrioritizedReplayBuffer):
    """
    Extends Static Buffer with N-Step Returns calculation tracking.
    Propagates deferred booking rewards backward n-steps efficiently.
    """
    def __init__(self, state_size, capacity=50000, n_step=3, gamma=0.99, **kwargs):
        super().__init__(state_size, capacity, **kwargs)
        self.n_step = n_step
        self.gamma = gamma
        self.n_step_buffer = []

    def push(self, state, action, reward, next_state, done):
        self.n_step_buffer.append((state, action, reward, next_state, done))
        
        if len(self.n_step_buffer) < self.n_step:
            if not done: return
            
        while self.n_step_buffer:
            reward_sum, n_state, n_done = self._calc_nstep()
            s, a, _, _, _ = self.n_step_buffer.pop(0)
            super().push(s, a, reward_sum, n_state, n_done)
            
            if not done and len(self.n_step_buffer) < self.n_step:
                break
                
    def _calc_nstep(self):
        reward_sum = 0
        for i, transition in enumerate(self.n_step_buffer):
            reward_sum += transition[2] * (self.gamma ** i)
            if transition[4]: # if done
                return reward_sum, transition[3], transition[4]
        return reward_sum, self.n_step_buffer[-1][3], self.n_step_buffer[-1][4]


class DQNAgent:
    """Enhanced Dueling DQN Agent with Soft Target Updates and N-Step Return support."""
    
    def __init__(self, state_size, action_size, 
                 learning_rate=0.0005, 
                 gamma=0.99, 
                 epsilon=1.0, 
                 epsilon_decay=0.995, 
                 epsilon_min=0.01, 
                 batch_size=64, 
                 hidden_size=256,
                 replay_buffer_size=50000,
                 use_prioritized_replay=True,
                 priority_alpha=0.6,
                 priority_beta=0.4,
                 priority_beta_increment=0.001,
                 gradient_clip=1.0,
                 learning_rate_decay=0.9,
                 lr_decay_step=200,
                 n_step=3,         # ← N-Step support
                 tau=0.005,        # ← Soft update factor
                 device=None):
        
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.n_step = n_step
        self.n_step_gamma = gamma ** n_step
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.use_prioritized_replay = use_prioritized_replay
        self.gradient_clip = gradient_clip
        self.lr_decay_step = lr_decay_step
        self.tau = tau
        
        if device is None:
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = device
        
        print(f"✓ Dueling DQN Agent initialized")
        print(f"  Device: {self.device}")
        
        # Networks (Dueling DQN)
        self.policy_net = DuelingDQNNetwork(state_size, action_size, hidden_size).to(self.device)
        self.target_net = DuelingDQNNetwork(state_size, action_size, hidden_size).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer, 
            step_size=lr_decay_step, 
            gamma=learning_rate_decay
        )
        self.criterion = nn.SmoothL1Loss(reduction='none') # Don't mean yet, apply weights first
        
        # Static Replay Buffer
        self.memory = NStepReplayBuffer(
            state_size=state_size,
            capacity=replay_buffer_size,
            n_step=n_step,
            gamma=gamma,
            alpha=priority_alpha,
            beta=priority_beta,
            beta_increment=priority_beta_increment
        )
        
        self.training_rewards = []
        self.losses = []
        self.episode_count = 0
        self.training_steps = 0
        
        self.action_names = {
            0: "E↓10% B↓10%", 1: "E↓10% B→",   2: "E↓10% B↑10%",
            3: "E→ B↓10%",    4: "E→ B→",      5: "E→ B↑10%",
            6: "E↑10% B↓10%", 7: "E↑10% B→",   8: "E↑10% B↑10%",
        }
        
    def select_action(self, state, training=True):
        if training and random.random() < self.epsilon:
            return random.randrange(self.action_size)
        
        with torch.no_grad():
            if not isinstance(state, np.ndarray):
                state = np.array(state, dtype=np.float32)
            
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.policy_net(state_tensor)
            return q_values.argmax().item()
    
    def store_transition(self, state, action, reward, next_state, done):
        self.memory.push(state, action, reward, next_state, done)
    
    def train_step(self):
        if len(self.memory) < self.batch_size:
            return None
        
        sample_result = self.memory.sample(self.batch_size)
        if sample_result is None: return None
        
        # Data is already perfectly shaped NumPy arrays
        states, actions, rewards, next_states, dones, indices, weights = sample_result
        
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        weights = torch.FloatTensor(weights).to(self.device)
        
        # Double DQN Calculation
        with torch.no_grad():
            # Action selection from Policy Net
            best_next_actions = self.policy_net(next_states).argmax(1, keepdim=True)
            # Action Evaluation from Target Net
            next_q_values = self.target_net(next_states).gather(1, best_next_actions).squeeze()
            
            # Use n_step_gamma to scale next_states properly
            target_q_values = rewards + (1 - dones) * self.n_step_gamma * next_q_values
            
        current_q_values = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze()
        
        td_errors = (current_q_values - target_q_values).abs().detach()

        loss = (weights * self.criterion(current_q_values, target_q_values)).mean()
        
        # Update Priorities
        self.memory.update_priorities(indices, td_errors.cpu().numpy())
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), self.gradient_clip)
        self.optimizer.step()
        self.scheduler.step()
        
        # Soft Update Target Network EVERY step
        self._soft_update_target_network()
        
        self.training_steps += 1
        return loss.item()
    
    def _soft_update_target_network(self):
        """Standard Polyak Averaging Soft Update"""
        for target_param, policy_param in zip(self.target_net.parameters(), self.policy_net.parameters()):
            target_param.data.copy_(self.tau * policy_param.data + (1.0 - self.tau) * target_param.data)
            
    def update_target_network(self):
        """Hard update fallback (rarely used, kept for compatibility)"""
        self.target_net.load_state_dict(self.policy_net.state_dict())
    
    def update_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
    
    def save_model(self, filepath, include_optimizer=True):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        checkpoint = {
            'policy_net': self.policy_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'epsilon': self.epsilon,
            'episode_count': self.episode_count,
            'training_steps': self.training_steps,
            'state_size': self.state_size,
            'action_size': self.action_size,
            'hidden_size': self.hidden_size,
        }
        if include_optimizer:
            checkpoint['optimizer'] = self.optimizer.state_dict()
            checkpoint['scheduler'] = self.scheduler.state_dict()
        torch.save(checkpoint, filepath)
        print(f"💾 Model saved to {filepath}")
    
    def load_model(self, filepath, load_optimizer=True):
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")
        checkpoint = torch.load(filepath, map_location=self.device, weights_only=False)
        self.policy_net.load_state_dict(checkpoint['policy_net'])
        self.target_net.load_state_dict(checkpoint['target_net'])
        self.epsilon = checkpoint.get('epsilon', self.epsilon_min)
        self.episode_count = checkpoint.get('episode_count', 0)
        self.training_steps = checkpoint.get('training_steps', 0)
        
        if load_optimizer and 'optimizer' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer'])
        if load_optimizer and 'scheduler' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler'])
        print(f"✓ Dueling Model loaded from {filepath}")
    
    def get_action_distribution(self, state):
        with torch.no_grad():
            if not isinstance(state, np.ndarray):
                state = np.array(state, dtype=np.float32)
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            return self.policy_net(state_tensor).cpu().numpy()[0]
    
    def get_best_action(self, state):
        with torch.no_grad():
            if not isinstance(state, np.ndarray):
                state = np.array(state, dtype=np.float32)
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.policy_net(state_tensor)
            action = q_values.argmax().item()
            return action, q_values.max().item(), self.action_names.get(action, f"Action {action}")
