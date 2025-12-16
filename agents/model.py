"""
Deep Q-Network (DQN) Agent for Airline Revenue Management
File: agents/model.py
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
import os


class DQNNetwork(nn.Module):
    """Deep Q-Network for airline pricing decisions"""
    
    def __init__(self, state_size, action_size, hidden_size=128):
        super(DQNNetwork, self).__init__()
        
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, action_size)
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.relu(self.fc3(x))
        x = self.fc4(x)
        return x


class ReplayBuffer:
    """Experience replay buffer for DQN"""
    
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        """Store a transition"""
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        """Sample a batch of transitions"""
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (np.array(states), np.array(actions), np.array(rewards), 
                np.array(next_states), np.array(dones))
    
    def __len__(self):
        return len(self.buffer)


class DQNAgent:
    """
    Deep Q-Learning Agent for Airline Revenue Management
    
    This agent learns optimal pricing strategies through interaction
    with the airline environment using deep reinforcement learning.
    """
    
    def __init__(self, state_size, action_size, learning_rate=0.001, 
                 gamma=0.99, epsilon=1.0, epsilon_decay=0.995, 
                 epsilon_min=0.01, batch_size=64, device=None):
        """
        Initialize DQN Agent
        
        Args:
            state_size: Dimension of state space
            action_size: Number of possible actions
            learning_rate: Learning rate for optimizer
            gamma: Discount factor for future rewards
            epsilon: Initial exploration rate
            epsilon_decay: Rate of epsilon decay
            epsilon_min: Minimum epsilon value
            batch_size: Size of training batches
            device: torch device (cpu/cuda)
        """
        
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size
        
        # Device setup
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        
        print(f"DQN Agent initialized on device: {self.device}")
        
        # Q-Networks
        self.policy_net = DQNNetwork(state_size, action_size).to(self.device)
        self.target_net = DQNNetwork(state_size, action_size).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        # Optimizer
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()
        
        # Replay buffer
        self.memory = ReplayBuffer(capacity=10000)
        
        # Training statistics
        self.training_rewards = []
        self.losses = []
        self.episode_count = 0
        
    def select_action(self, state, training=True):
        """
        Select action using epsilon-greedy policy
        
        Args:
            state: Current state observation
            training: If True, use epsilon-greedy; if False, use greedy
            
        Returns:
            action: Selected action index
        """
        if training and random.random() < self.epsilon:
            # Exploration: random action
            return random.randrange(self.action_size)
        
        # Exploitation: best action from Q-network
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.policy_net(state_tensor)
            return q_values.argmax().item()
    
    def store_transition(self, state, action, reward, next_state, done):
        """Store transition in replay buffer"""
        self.memory.push(state, action, reward, next_state, done)
    
    def train_step(self):
        """
        Perform one training step (batch update)
        
        Returns:
            loss: Training loss value or None if buffer too small
        """
        if len(self.memory) < self.batch_size:
            return None
        
        # Sample batch from replay buffer
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        
        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        # Current Q values: Q(s, a)
        current_q_values = self.policy_net(states).gather(1, actions.unsqueeze(1))
        
        # Next Q values: max_a' Q(s', a')
        with torch.no_grad():
            next_q_values = self.target_net(next_states).max(1)[0]
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
        
        # Compute loss
        loss = self.criterion(current_q_values.squeeze(), target_q_values)
        
        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        
        self.optimizer.step()
        
        return loss.item()
    
    def update_target_network(self):
        """Update target network with policy network weights"""
        self.target_net.load_state_dict(self.policy_net.state_dict())
    
    def update_epsilon(self):
        """Decay epsilon for exploration-exploitation tradeoff"""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
    
    def save_model(self, filepath, include_optimizer=True):
        """
        Save model weights and training state
        
        Args:
            filepath: Path to save the model
            include_optimizer: Whether to save optimizer state
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        checkpoint = {
            'policy_net': self.policy_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'epsilon': self.epsilon,
            'episode_count': self.episode_count,
            'training_rewards': self.training_rewards,
            'losses': self.losses
        }
        
        if include_optimizer:
            checkpoint['optimizer'] = self.optimizer.state_dict()
        
        torch.save(checkpoint, filepath)
        print(f"✓ Model saved to {filepath}")
    
    def load_model(self, filepath, load_optimizer=True):
        """
        Load model weights and training state
        
        Args:
            filepath: Path to load the model from
            load_optimizer: Whether to load optimizer state
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")
        
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.policy_net.load_state_dict(checkpoint['policy_net'])
        self.target_net.load_state_dict(checkpoint['target_net'])
        self.epsilon = checkpoint.get('epsilon', self.epsilon_min)
        self.episode_count = checkpoint.get('episode_count', 0)
        self.training_rewards = checkpoint.get('training_rewards', [])
        self.losses = checkpoint.get('losses', [])
        
        if load_optimizer and 'optimizer' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer'])
        
        print(f"✓ Model loaded from {filepath}")
        print(f"  Episodes trained: {self.episode_count}")
        print(f"  Current epsilon: {self.epsilon:.4f}")
    
    def get_action_distribution(self, state):
        """
        Get Q-values for all actions (for visualization/analysis)
        
        Args:
            state: Current state observation
            
        Returns:
            q_values: Array of Q-values for each action
        """
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.policy_net(state_tensor)
            return q_values.cpu().numpy()[0]
    
    def get_best_action(self, state):
        """
        Get the best action without exploration
        
        Args:
            state: Current state observation
            
        Returns:
            action: Best action index
            q_value: Q-value of the best action
        """
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.policy_net(state_tensor)
            action = q_values.argmax().item()
            q_value = q_values.max().item()
            return action, q_value


# Example usage and testing
if __name__ == "__main__":
    print("="*70)
    print("  DQN AGENT - MODEL TESTING")
    print("="*70)
    
    # Initialize agent
    state_size = 7  # As defined in airline environment
    action_size = 5  # 5 pricing actions
    
    agent = DQNAgent(state_size, action_size)
    
    # Test action selection
    print("\n1. Testing Action Selection:")
    test_state = np.array([6000, 5950, 45, 0.7, 0.6, 0, 14], dtype=np.float32)
    
    action = agent.select_action(test_state, training=True)
    print(f"   Selected action (with exploration): {action}")
    
    best_action, q_value = agent.get_best_action(test_state)
    print(f"   Best action (greedy): {best_action} (Q={q_value:.2f})")
    
    # Test Q-values
    print("\n2. Q-values for all actions:")
    q_values = agent.get_action_distribution(test_state)
    action_names = ['-20%', '-10%', '0%', '+10%', '+20%']
    for i, (name, q) in enumerate(zip(action_names, q_values)):
        print(f"   Action {i} ({name:>5}): Q = {q:>8.4f}")
    
    # Test memory operations
    print("\n3. Testing Replay Buffer:")
    for i in range(100):
        state = np.random.rand(state_size)
        action = np.random.randint(action_size)
        reward = np.random.rand()
        next_state = np.random.rand(state_size)
        done = i % 20 == 0
        agent.store_transition(state, action, reward, next_state, done)
    
    print(f"   Buffer size: {len(agent.memory)}/10000")
    
    # Test training step
    print("\n4. Testing Training Step:")
    loss = agent.train_step()
    if loss:
        print(f"   Training loss: {loss:.4f}")
    
    # Test save/load
    print("\n5. Testing Save/Load:")
    test_path = "models/test_model.pth"
    agent.save_model(test_path)
    
    # Create new agent and load
    new_agent = DQNAgent(state_size, action_size)
    new_agent.load_model(test_path)
    
    # Cleanup
    if os.path.exists(test_path):
        os.remove(test_path)
        print("   Test model file cleaned up")
    
    print("\n" + "="*70)
    print("  ✓ ALL TESTS PASSED")
    print("="*70)