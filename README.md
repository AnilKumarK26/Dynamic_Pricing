# RL-Integrated Multi-Class Airline Dynamic Pricing Dashboard

This project implements a Reinforcement Learning (RL) based Revenue Management (RM) system for airlines. It uses a Deep Q-Network (DQN) agent with Prioritized Experience Replay (PER) to optimize pricing for multiple seat classes (Economy and Business) across various routes, calibrated with real flight data.

## 🚀 Project Overview

The system simulates a competitive airline market where an RL agent learns to adjust prices dynamically based on:
- **Days remaining** until departure (90-day horizon).
- **Current occupancy** (Load Factor) for both Economy and Business classes.
- **Competitor pricing strategies** (Dynamic market response).
- **Market disruptions** (Weather, pilot strikes, competitor cancellations).
- **Route-specific demand** (Calibrated from historical data).

## ✨ Key Features

- **Multi-Class Optimization**: Jointly optimizes Economy and Business class pricing using a 9-action discrete space.
- **Multi-Route Generalization**: Agent trains across dozens of routes, learning shared market dynamics while respecting route-specific nuances.
- **Revenue-Dominant Reward (v3)**: A refined reward function that prioritizes total revenue (90% signal) over simple occupancy, ensuring the agent learns to maximize profit rather than just filling seats.
- **Curriculum Learning**: Training follows a 3-phase curriculum:
  1. **Phase 1**: High-traffic "easy" routes to learn basic pricing logic.
  2. **Phase 2**: Expanded route set for broader generalization.
  3. **Phase 3**: Full multi-route mastery across the entire network.
- **Advanced RL Architecture**: DQN with LayerNorm, Dropout, and Prioritized Experience Replay for stable and efficient learning.
- **Interactive Dashboard**: Real-time simulation control, AI recommendations, and automated strategy comparison (RL vs. Rule-based vs. Random).

## 📂 Folder Structure

```text
Dynamic_Pricing/
├── agents/             # DQN Agent implementation with PER
├── baselines/          # Traditional pricing strategies (Rule-based, Random, etc.)
├── config/             # Centralized Hyperparameters & Curriculum settings
├── data/               # Input CSVs and calibrated route statistics (.pkl)
├── environment/        # Gymnasium-based Airline Revenue Environment (v3)
├── models/             # Storage for trained models and checkpoints
│   └── trained_models/ # Best and final model checkpoints (.pth)
├── results/            # Training visualizations, evaluation logs, and JSON stats
├── static/             # Frontend assets (Tailwind-like CSS, JS, Lottie)
├── templates/          # HTML templates for the Flask dashboard
├── training/           # Training pipeline with Curriculum Learning support
├── utils/              # Data preprocessing and helper utilities
├── analyze_data.py     # Environment calibration script
├── app.py              # Main Flask application (Dashboard & API)
├── dockerfile          # Container build (uses uv for dependencies)
├── docker-compose.yml  # Run app via: docker compose up --build
├── requirements.txt    # Python dependencies
├── setup.py            # Project initialization script
└── run.sh              # One-time setup: structure, deps (uv), calibration
```

## 🛠️ Setup and Installation

**Quick path:** Run `./run.sh` once—it does steps 1–3 below (structure, uv venv + deps, calibration). Then use **Docker Compose** or `python3 app.py` to start the app (see [Running the Project](#-running-the-project)).

**Manual setup:**

### 1. Initialize Project
```bash
python setup.py
```

### 2. Install Dependencies (Using uv)
We recommend [uv](https://github.com/astral-sh/uv) for fast package management:
```bash
uv venv
source .venv/bin/activate   # On Windows: .venv\Scripts\activate
uv pip install -r requirements.txt
```

### 3. Calibrate Environment
Put flight data in `data/flight_data.csv` (or use `data/sample_data.csv`). Then run:
```bash
python analyze_data.py
```
This generates `data/route_stats.pkl`, which the RL environment uses for route-specific demand and competitor behavior.

## 🏃 Running the Project

### Recommended: Setup with run.sh, then run with Docker Compose
1. **One-time setup** (project structure, dependencies via uv, data calibration):
   ```bash
   ./run.sh
   ```
   This creates `data/route_stats.pkl` and prints the next commands.

2. **Run the app** with Docker Compose:
   ```bash
   docker compose up --build
   ```
   Or in the background: `docker compose up --build -d`  
   Open [http://localhost:8080](http://localhost:8080) in your browser.

### Run locally (no Docker)
```bash
./run.sh        # if you haven't set up yet
python3 app.py
```
Open [http://localhost:8080](http://localhost:8080) in your browser.

### Train the RL Agent
To train the model from scratch using the Curriculum Learning strategy:
```bash
python training/train.py
```
*Note: Default training runs for 6000 episodes as per `config.py`.*

## 📡 API Endpoints

- `GET /api/state`: Current simulation state (prices, load, competitors).
- `GET /api/ai_recommendation`: RL agent's best action with "reasoning" context.
- `POST /api/run_comparison`: Batch evaluation of RL vs Traditional strategies.
- `POST /api/action`: Manually execute a pricing action.
- `POST /api/change_route`: Switch simulation to a different calibrated route.

## 🐋 Docker

The **dockerfile** uses [uv](https://github.com/astral-sh/uv) to install dependencies for faster builds. Run after `./run.sh` setup (see **Running the Project** above).

**With Docker Compose (recommended):**
```bash
docker compose up --build
# Or detached:  docker compose up --build -d
```
Then open [http://localhost:8080](http://localhost:8080). Stop with `Ctrl+C` or `docker compose down`.

**Plain Docker:**
```bash
docker build -t dynamic-pricing .
docker run -p 8080:8080 dynamic-pricing
```
Rebuild when you change code or data.
