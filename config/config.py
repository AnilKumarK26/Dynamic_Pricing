"""
Configuration file for Airline RL Project
File: config/config.py
"""

import os
from pathlib import Path

# Base directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Data directories
DATA_DIR = BASE_DIR / 'data'
MODELS_DIR = BASE_DIR / 'models'
LOGS_DIR = BASE_DIR / 'logs'
RESULTS_DIR = BASE_DIR / 'results'

# Ensure directories exist
for dir_path in [DATA_DIR, MODELS_DIR, LOGS_DIR, RESULTS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# File paths
FLIGHT_DATA_PATH = DATA_DIR / 'flight_data.csv'
SAMPLE_DATA_PATH = DATA_DIR / 'sample_data.csv'

# Environment Configuration
ENV_CONFIG = {
    'total_seats': 180,
    'max_days': 90,
    'base_price': 6000,
    'default_route': 'Delhi-Mumbai',
}

# DQN Agent Configuration
AGENT_CONFIG = {
    'state_size': 7,
    'action_size': 5,
    'learning_rate': 0.001,
    'gamma': 0.99,
    'epsilon': 1.0,
    'epsilon_decay': 0.995,
    'epsilon_min': 0.01,
    'batch_size': 64,
    'hidden_size': 128,
    'replay_buffer_size': 10000,
}

# Training Configuration
TRAINING_CONFIG = {
    'num_episodes': 1000,
    'target_update_freq': 10,
    'save_freq': 100,
    'eval_episodes': 10,
    'max_steps_per_episode': 90,
}

# Flask Configuration
FLASK_CONFIG = {
    'host': '0.0.0.0',
    'port': 5000,
    'debug': True,
    'secret_key': 'airline_rl_secret_key_2024_secure',
}

# Logging Configuration
LOGGING_CONFIG = {
    'level': 'INFO',
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'log_file': LOGS_DIR / 'app.log',
}

# Action Names (for display)
ACTION_NAMES = {
    0: 'Decrease 20%',
    1: 'Decrease 10%',
    2: 'Hold Price',
    3: 'Increase 10%',
    4: 'Increase 20%',
}

ACTION_SYMBOLS = {
    0: '📉 -20%',
    1: '📉 -10%',
    2: '⏸️ 0%',
    3: '📈 +10%',
    4: '📈 +20%',
}

# Price adjustment multipliers
PRICE_ADJUSTMENTS = {
    0: -0.20,
    1: -0.10,
    2: 0.00,
    3: 0.10,
    4: 0.20,
}

# Disruption types
DISRUPTION_TYPES = {
    'none': {
        'name': 'Normal Operations',
        'demand_factor': 1.0,
        'description': 'Business as usual'
    },
    'weather': {
        'name': 'Weather Delay',
        'demand_factor': 0.6,
        'description': 'Bad weather affecting flights'
    },
    'pilot_strike': {
        'name': 'Pilot Strike',
        'demand_factor': 0.3,
        'description': 'Industrial action by pilots'
    },
    'competitor_cancel': {
        'name': 'Competitor Cancellation',
        'demand_factor': 1.5,
        'description': 'Competitor cancelled flights'
    }
}

# Airlines (for competitor simulation)
AIRLINES = ['SpiceJet', 'AirAsia', 'Vistara', 'IndiGo']

# Routes
ROUTES = [
    'Delhi-Mumbai',
    'Delhi-Bangalore',
    'Mumbai-Chennai',
    'Mumbai-Kolkata',
    'Delhi-Hyderabad',
]

# Model save path template
MODEL_SAVE_PATH = str(MODELS_DIR / 'trained_models' / 'model_ep{episode}.pth')
BEST_MODEL_PATH = str(MODELS_DIR / 'best_model.pth')
FINAL_MODEL_PATH = str(MODELS_DIR / 'final_model.pth')

# Results save paths
TRAINING_STATS_PATH = str(RESULTS_DIR / 'training_stats.json')
TRAINING_PLOT_PATH = str(RESULTS_DIR / 'training_progress.png')
EVALUATION_RESULTS_PATH = str(RESULTS_DIR / 'evaluation_results.json')


def get_config():
    """Get all configuration as a dictionary"""
    return {
        'env': ENV_CONFIG,
        'agent': AGENT_CONFIG,
        'training': TRAINING_CONFIG,
        'flask': FLASK_CONFIG,
        'logging': LOGGING_CONFIG,
    }


def print_config():
    """Print current configuration"""
    print("="*70)
    print("  AIRLINE RL PROJECT CONFIGURATION")
    print("="*70)
    
    print("\n📁 Directories:")
    print(f"   Base: {BASE_DIR}")
    print(f"   Data: {DATA_DIR}")
    print(f"   Models: {MODELS_DIR}")
    print(f"   Logs: {LOGS_DIR}")
    print(f"   Results: {RESULTS_DIR}")
    
    print("\n🌍 Environment:")
    for key, value in ENV_CONFIG.items():
        print(f"   {key}: {value}")
    
    print("\n🤖 Agent:")
    for key, value in AGENT_CONFIG.items():
        print(f"   {key}: {value}")
    
    print("\n🎓 Training:")
    for key, value in TRAINING_CONFIG.items():
        print(f"   {key}: {value}")
    
    print("\n🌐 Flask:")
    for key, value in FLASK_CONFIG.items():
        if key != 'secret_key':
            print(f"   {key}: {value}")
    
    print("\n" + "="*70)


if __name__ == "__main__":
    print_config()