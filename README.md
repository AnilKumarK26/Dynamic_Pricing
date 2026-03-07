# RL-Integrated Multi-Class Airline Dynamic Pricing Dashboard

This project implements a Reinforcement Learning (RL) based Revenue Management (RM) system for airlines. It uses a Deep Q-Network (DQN) agent to optimize pricing for multiple seat classes (Economy and Business) across various routes, calibrated with real flight data.

## 🚀 Project Overview

The system simulates a competitive airline market where an RL agent learns to adjust prices dynamically based on:
- Days remaining until departure.
- Current seat occupancy (load factor).
- Competitor pricing strategies.
- Market disruptions (weather, strikes, competitor cancellations).

## 📋 Prerequisites

- **Python**: 3.10.x (recommended)
- **Pip**: Python package manager
- **Dependencies**: Listed in `requirements.txt` (Flask, Torch, NumPy, Pandas, etc.)

## 📂 Folder Structure

```text
Dynamic_Pricing/
├── agents/             # RL Agent implementations (DQN)
├── baselines/          # Traditional pricing strategies for comparison
├── config/             # Project and Hyperparameter configurations
├── data/               # Input data (CSV) and calibrated statistics (.pkl)
├── environment/        # Gymnasium-based Airline Revenue Environment
├── models/             # Storage for trained models (.pth)
│   └── trained_models/ # Best and final model checkpoints
├── results/            # Data visualizations and evaluation logs
├── static/             # Frontend assets (CSS, JS, Images)
├── templates/          # HTML templates for the Flask dashboard
├── training/           # Training scripts and pipelines
├── utils/              # Data preprocessing and helper utilities
├── analyze_data.py     # Script to calibrate environment from CSV data
├── app.py              # Main Flask application (Dashboard)
├── dockerfile          # Containerization configuration
├── requirements.txt    # Python dependencies
├── setup.py            # Project initialization script
└── run.sh              # (New) Automated startup script
```

## 🛠️ Setup and Installation

### 1. Initialize Project Structure
Run the setup script to create necessary directories:
```bash
python setup.py
```
*Note: You can choose to generate sample data during this step.*

### 2. Install Dependencies (Using uv)
We recommend using [uv](https://github.com/astral-sh/uv) for fast package management:
```bash
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

### 3. Prepare and Calibrate Data
Ensure your flight data is in `data/flight_data.csv`. Then run:
```bash
python analyze_data.py
```
This generates `data/route_stats.pkl`, which is required for the RL environment.

## 🏃 Running the Project

### Automated Startup (Recommended)
The easiest way to run the project is using the provided script. **It will automatically install `uv` and set up the virtual environment for you**:
```bash
./run.sh
```

### Manual Setup (Using uv)
If you prefer to setup manually:
```bash
# Install uv if missing (https://github.com/astral-sh/uv)
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt
python3 analyze_data.py
python3 app.py
```
Open your browser at [http://localhost:5000](http://localhost:5000).

### Train the RL Agent
To train the model from scratch:
```bash
python training/train.py
```

## 📡 API Endpoints

- `GET /api/state`: Current simulation state.
- `GET /api/ai_recommendation`: Get best pricing action from the RL agent.
- `POST /api/run_comparison`: Compare RL vs traditional strategies.
- `POST /api/change_route`: Switch to a different flight route.

## 💻 OS Support

- **macOS / Linux**: Fully supported via the `./run.sh` script.
- **Windows**: 
  - The core logic is the same, but the `.sh` script may not run natively in CMD/PowerShell.
  - Windows users should follow the **Manual Setup** steps using `pip install uv` first.
  - Activation command on Windows: `.venv\Scripts\activate` (instead of `source .venv/bin/activate`).

## 🐋 Docker Support
To run using Docker (cross-platform):
```bash
docker build -t dynamic-pricing .
docker run -p 5000:5000 dynamic-pricing
```
