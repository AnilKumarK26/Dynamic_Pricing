import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import json
from datetime import datetime

# Import custom modules (assuming they're in the same directory or installed)
# from environment.airline_env import AirlineRevenueEnv
# from agents.model import DQNAgent
# from utils.preprocessing import FlightDataProcessor


class AirlineRLTrainer:
    """Training pipeline for Airline RL Agent"""
    
    def __init__(self, env, agent, save_dir='models/'):
        self.env = env
        self.agent = agent
        self.save_dir = save_dir
        
        # Training statistics
        self.episode_rewards = []
        self.episode_revenues = []
        self.episode_load_factors = []
        self.episode_lengths = []
        self.losses = []
        
    def train(self, num_episodes=1000, target_update_freq=10, 
              save_freq=100, verbose=True):
        """Train the DQN agent"""
        
        print(f"Starting training for {num_episodes} episodes...")
        print(f"Device: {self.agent.device}")
        
        for episode in tqdm(range(num_episodes), desc="Training"):
            state = self.env.reset()
            episode_reward = 0
            episode_loss = []
            done = False
            step = 0
            
            while not done:
                # Select action
                action = self.agent.select_action(state, training=True)
                
                # Take action
                next_state, reward, done, info = self.env.step(action)
                
                # Store transition
                self.agent.store_transition(state, action, reward, next_state, done)
                
                # Train agent
                loss = self.agent.train_step()
                if loss is not None:
                    episode_loss.append(loss)
                
                episode_reward += reward
                state = next_state
                step += 1
            
            # Update target network
            if episode % target_update_freq == 0:
                self.agent.update_target_network()
            
            # Decay epsilon
            self.agent.update_epsilon()
            
            # Store statistics
            self.episode_rewards.append(episode_reward)
            self.episode_revenues.append(info['total_revenue'])
            self.episode_load_factors.append(info['seats_sold'] / self.env.total_seats)
            self.episode_lengths.append(step)
            if episode_loss:
                self.losses.append(np.mean(episode_loss))
            
            # Verbose logging
            if verbose and episode % 50 == 0:
                avg_reward = np.mean(self.episode_rewards[-50:])
                avg_revenue = np.mean(self.episode_revenues[-50:])
                avg_load = np.mean(self.episode_load_factors[-50:])
                print(f"\nEpisode {episode}/{num_episodes}")
                print(f"  Avg Reward (last 50): {avg_reward:.2f}")
                print(f"  Avg Revenue: ₹{avg_revenue:.2f}")
                print(f"  Avg Load Factor: {avg_load*100:.1f}%")
                print(f"  Epsilon: {self.agent.epsilon:.3f}")
                if self.losses:
                    print(f"  Avg Loss: {np.mean(self.losses[-50:]):.4f}")
            
            # Save model
            if episode % save_freq == 0 and episode > 0:
                self.save_checkpoint(episode)
        
        print("\n✓ Training completed!")
        self.save_final_model()
        self.save_training_stats()
        
    def evaluate(self, num_episodes=10, render=False):
        """Evaluate trained agent"""
        print(f"\nEvaluating agent for {num_episodes} episodes...")
        
        eval_rewards = []
        eval_revenues = []
        eval_load_factors = []
        
        for episode in range(num_episodes):
            state = self.env.reset()
            episode_reward = 0
            done = False
            
            while not done:
                # Select action (no exploration)
                action = self.agent.select_action(state, training=False)
                next_state, reward, done, info = self.env.step(action)
                
                if render:
                    self.env.render()
                
                episode_reward += reward
                state = next_state
            
            eval_rewards.append(episode_reward)
            eval_revenues.append(info['total_revenue'])
            eval_load_factors.append(info['seats_sold'] / self.env.total_seats)
            
            if render:
                print(f"\n--- Episode {episode+1} Summary ---")
                print(f"Total Reward: {episode_reward:.2f}")
                print(f"Total Revenue: ₹{info['total_revenue']:.2f}")
                print(f"Load Factor: {(info['seats_sold']/self.env.total_seats)*100:.1f}%")
        
        # Print evaluation results
        print("\n=== Evaluation Results ===")
        print(f"Average Reward: {np.mean(eval_rewards):.2f} ± {np.std(eval_rewards):.2f}")
        print(f"Average Revenue: ₹{np.mean(eval_revenues):.2f} ± {np.std(eval_revenues):.2f}")
        print(f"Average Load Factor: {np.mean(eval_load_factors)*100:.1f}% ± {np.std(eval_load_factors)*100:.1f}%")
        
        return {
            'rewards': eval_rewards,
            'revenues': eval_revenues,
            'load_factors': eval_load_factors
        }
    
    def save_checkpoint(self, episode):
        """Save training checkpoint"""
        filepath = f"{self.save_dir}checkpoint_ep{episode}.pth"
        self.agent.save_model(filepath)
    
    def save_final_model(self):
        """Save final trained model"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = f"{self.save_dir}final_model_{timestamp}.pth"
        self.agent.save_model(filepath)
    
    def save_training_stats(self):
        """Save training statistics"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        stats = {
            'episode_rewards': self.episode_rewards,
            'episode_revenues': self.episode_revenues,
            'episode_load_factors': self.episode_load_factors,
            'episode_lengths': self.episode_lengths,
            'losses': self.losses
        }
        
        filepath = f"{self.save_dir}training_stats_{timestamp}.json"
        with open(filepath, 'w') as f:
            json.dump(stats, f)
        print(f"Training stats saved to {filepath}")
    
    def plot_training_progress(self, save_path=None):
        """Plot training metrics"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot rewards
        axes[0, 0].plot(self.episode_rewards, alpha=0.3, label='Episode Reward')
        axes[0, 0].plot(self._moving_average(self.episode_rewards, 50), 
                       label='MA(50)', linewidth=2)
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Reward')
        axes[0, 0].set_title('Training Rewards')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Plot revenue
        axes[0, 1].plot(self.episode_revenues, alpha=0.3, label='Episode Revenue')
        axes[0, 1].plot(self._moving_average(self.episode_revenues, 50), 
                       label='MA(50)', linewidth=2)
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].set_ylabel('Revenue (₹)')
        axes[0, 1].set_title('Total Revenue per Episode')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Plot load factor
        load_factors_pct = [lf * 100 for lf in self.episode_load_factors]
        axes[1, 0].plot(load_factors_pct, alpha=0.3, label='Load Factor')
        axes[1, 0].plot(self._moving_average(load_factors_pct, 50), 
                       label='MA(50)', linewidth=2)
        axes[1, 0].set_xlabel('Episode')
        axes[1, 0].set_ylabel('Load Factor (%)')
        axes[1, 0].set_title('Load Factor per Episode')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # Plot loss
        if self.losses:
            axes[1, 1].plot(self.losses, alpha=0.5)
            axes[1, 1].plot(self._moving_average(self.losses, 50), 
                           linewidth=2, label='MA(50)')
            axes[1, 1].set_xlabel('Episode')
            axes[1, 1].set_ylabel('Loss')
            axes[1, 1].set_title('Training Loss')
            axes[1, 1].set_yscale('log')
            axes[1, 1].legend()
            axes[1, 1].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Training plot saved to {save_path}")
        
        plt.show()
    
    @staticmethod
    def _moving_average(data, window):
        """Calculate moving average"""
        if len(data) < window:
            return data
        return np.convolve(data, np.ones(window)/window, mode='valid')


# Example training script
if __name__ == "__main__":
    print("=== Airline RL Training Pipeline ===\n")
    
    # 1. Load and preprocess data
    print("1. Loading flight data...")
    # processor = FlightDataProcessor()
    # df = processor.load_data('data/flight_data.csv')
    # df_processed = processor.preprocess(df)
    
    # For demo purposes, create dummy data
    dummy_data = pd.DataFrame({
        'route': ['Delhi-Mumbai'] * 50,
        'airline': ['SpiceJet', 'AirAsia', 'Vistara', 'IndiGo', 'GoAir'] * 10,
        'price': np.random.normal(6000, 200, 50)
    })
    
    # 2. Create environment
    print("2. Creating RL environment...")
    # env = AirlineRevenueEnv(df_processed, route='Delhi-Mumbai')
    
    # For demo
    print("   (Using dummy environment for demo)")
    
    # 3. Create agent
    print("3. Initializing DQN agent...")
    state_size = 7
    action_size = 5
    # agent = DQNAgent(state_size, action_size)
    
    # 4. Create trainer
    print("4. Setting up trainer...")
    # trainer = AirlineRLTrainer(env, agent)
    
    # 5. Train
    print("\n5. Starting training...")
    print("   [This would train for 1000 episodes]")
    # trainer.train(num_episodes=1000, verbose=True)
    
    # 6. Evaluate
    print("\n6. Evaluating trained agent...")
    # results = trainer.evaluate(num_episodes=10, render=True)
    
    # 7. Plot results
    print("\n7. Plotting training progress...")
    # trainer.plot_training_progress(save_path='training_progress.png')
    
    print("\n✓ Training pipeline setup complete!")
    print("\nTo run actual training, uncomment the code sections and ensure")
    print("all dependencies (environment, agent, data) are properly set up.")