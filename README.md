# Dynamic Pricing for Airline Revenue Management

This repository contains an airline revenue management project that uses reinforcement learning and traditional pricing baselines to simulate route-level pricing decisions.

## Project Overview

- `app.py` - Flask dashboard and API for running the airline pricing simulation.
- `agents/model.py` - RL agent implementation (DQN-based pricing policy).
- `environment/airline_env.py` - Custom environment modeling airline revenue and demand.
- `baselines/traditional_pricing.py` - Baseline pricing strategies for comparison.
- `training/train.py` - Training pipeline for RL agents.
- `utils/preprocessing.py` - Data preparation utilities.
- `config/config.py` - Configuration values and state size computation.
- `data/flight_data.csv` - Flight dataset used for training and evaluation.
- `models/trained_models/` - Saved model checkpoints and trained weights.
- `results/` - Evaluation logs and training statistics.
- `templates/` and `static/` - Web dashboard frontend assets.

## Requirements

Install required packages with:

```bash
pip install -r requirements.txt
```

## Running the Project

1. Ensure Python and required dependencies are installed.
2. Place or verify your dataset in `data/flight_data.csv`.
3. Start the Flask application:

```bash
python app.py
```

4. Open a browser and navigate to:

```text
http://localhost:5000
```

## Training

Training can be launched from the `training/train.py` module. This project uses an RL environment and DQN agent to learn pricing actions over simulated flight demand.

## Notes

- The app tries to load a trained model from `models/trained_models/`.
- If no trained model is available, the dashboard can still run using an untrained policy.
- `config/config.py` contains hyperparameters and state-size computation logic.

## Project Structure

- `agents/` - Reinforcement learning agent code.
- `baselines/` - Traditional pricing strategy implementations.
- `config/` - Configuration settings.
- `data/` - Dataset files.
- `environment/` - Custom gymnasium environment.
- `models/` - Saved model artifacts.
- `results/` - Logs and evaluation output.
- `static/`, `templates/` - Dashboard UI.
- `training/` - Training scripts.
- `utils/` - Helper functions and preprocessing.

## License

This repository does not include a license file. Add one if you plan to share or publish the project.
