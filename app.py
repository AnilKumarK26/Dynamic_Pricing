"""
FastAPI Dashboard for RL-based Airline Revenue Management
File: app.py
"""

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import numpy as np
import torch
import os
import pickle
import sys
import time
from typing import Optional, Dict, Any

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agents.model import DQNAgent
from environment.airline_env import AirlineRevenueEnv
from config.config import AGENT_CONFIG, FLASK_CONFIG, compute_state_size
from baselines.traditional_pricing import TRADITIONAL_STRATEGIES, compare_all_strategies

app = FastAPI(title="Airline RL Dashboard")

# Mount templates and static files
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

rl_agent        = None
rl_env          = None
agent_loaded    = False
comparison_results = None

_rec_cache      = {}
_rec_cache_time = 0

CALIBRATION_PATH = 'data/route_stats.pkl'


# ═══════════════════════════════════════════════════════════════════════════════
# RL SIMULATION STATE WRAPPER
# ═══════════════════════════════════════════════════════════════════════════════

class RLSimulationState:
    """Thin wrapper around the live AirlineRevenueEnv for the dashboard."""

    def __init__(self, env):
        self.env           = env
        self.calibrated    = True
        self.current_state = None
        self.done          = False

    def reset(self):
        state, info     = self.env.reset()
        self.current_state = state
        self.done          = False
        return self.current_state

    def step(self, action):
        next_state, reward, terminated, truncated, info = self.env.step(action)
        done               = terminated or truncated
        self.current_state = next_state
        self.done          = done
        return next_state, reward, done, info

    def get_state_dict(self):
        return {
            'route':           self.env.route,
            # Economy
            'econ_price':      float(self.env.econ_price),
            'econ_sold':       int(self.env.econ_sold),
            'econ_total':      int(self.env.econ_seats_total),
            'econ_load_factor': float(self.env.econ_sold / self.env.econ_seats_total * 100),
            'econ_revenue':    float(self.env.revenue_econ),
            # Business
            'bus_price':       float(self.env.bus_price),
            'bus_sold':        int(self.env.bus_sold),
            'bus_total':       int(self.env.bus_seats_total),
            'bus_load_factor': float(self.env.bus_sold / self.env.bus_seats_total * 100),
            'bus_revenue':     float(self.env.revenue_bus),
            # Overall
            'total_seats':     int(self.env.total_seats),
            'total_sold':      int(self.env.econ_sold + self.env.bus_sold),
            'load_factor':     float((self.env.econ_sold + self.env.bus_sold) / self.env.total_seats * 100),
            'total_revenue':   float(self.env.total_revenue),
            'days_to_departure': int(self.env.days_to_departure),
            'disruption':      self.env.current_disruption,
            # Competitors
            'econ_competitors': {k: float(v) for k, v in self.env.econ_competitors.items()},
            'bus_competitors':  {k: float(v) for k, v in self.env.bus_competitors.items()},
            'step':             int(self.env.current_step),
            'calibrated':       True,
            'available_routes': self.env.routes,
            'current_route':    self.env.route,
        }

sim_state = None


# ═══════════════════════════════════════════════════════════════════════════════
# SYSTEM LOADER
# ═══════════════════════════════════════════════════════════════════════════════

def load_rl_system():
    """Load RL environment and trained agent."""
    global rl_agent, rl_env, sim_state, agent_loaded

    print("\n" + "=" * 80)
    print("  🤖 LOADING RL SYSTEM")
    print("=" * 80)

    if not os.path.exists(CALIBRATION_PATH):
        print(f"\n❌ ERROR: No calibration file at {CALIBRATION_PATH}")
        print(f"   Run: python analyze_data.py")
        return False

    try:
        with open(CALIBRATION_PATH, 'rb') as f:
            route_stats = pickle.load(f)

        print(f"✓ Loaded calibration for {len(route_stats)} routes")

        rl_env = AirlineRevenueEnv(
            route_stats_path=CALIBRATION_PATH,
            fixed_route=None,
        )

        rl_env._route_stats_path = CALIBRATION_PATH

        print(f"✓ Created RL environment")
        print(f"  State space:  {rl_env.observation_space.shape[0]}")
        print(f"  Action space: {rl_env.action_space.n}")

        state_size = rl_env.observation_space.shape[0]
        AGENT_CONFIG['state_size'] = state_size

        rl_agent = DQNAgent(
            state_size=state_size,
            action_size=9,
            **{k: v for k, v in AGENT_CONFIG.items()
               if k not in ['state_size', 'action_size']},
        )
        print(f"✓ Created DQN agent")

        model_paths = [
            'models/trained_models/best_model.pth',
            'models/trained_models/final_model.pth',
        ]
        models_dir = 'models/trained_models'
        if os.path.exists(models_dir):
            for file in sorted(os.listdir(models_dir), reverse=True):
                if file.startswith('final_model_') and file.endswith('.pth'):
                    model_paths.insert(0, os.path.join(models_dir, file))
                    break

        model_loaded = False
        for model_path in model_paths:
            if os.path.exists(model_path):
                try:
                    rl_agent.load_model(model_path, load_optimizer=False)
                    rl_agent.epsilon = 0.0
                    print(f"✓ Loaded trained model: {model_path}")
                    model_loaded  = True
                    agent_loaded  = True
                    break
                except Exception as e:
                    print(f"⚠️  Failed to load {model_path}: {e}")

        if not model_loaded:
            print(f"\n⚠️  WARNING: No trained model found! Agent will use untrained policy.")
            agent_loaded = False

        sim_state = RLSimulationState(rl_env)
        sim_state.reset()

        print(f"\n✓ RL System Ready!")
        print(f"  Agent: {'TRAINED' if model_loaded else 'UNTRAINED'}")
        print("=" * 80)
        return True

    except Exception as e:
        print(f"\n❌ ERROR loading RL system: {e}")
        import traceback
        traceback.print_exc()
        return False

# Initialize during startup
@app.on_event("startup")
async def startup_event():
    global rl_system_loaded
    rl_system_loaded = load_rl_system()

# ═══════════════════════════════════════════════════════════════════════════════
# MODELS
# ═══════════════════════════════════════════════════════════════════════════════
class ChangeRouteRequest(BaseModel):
    route: str

class ActionRequest(BaseModel):
    action: int = 4

class DisruptionRequest(BaseModel):
    type: str = 'none'

class ResetRequest(BaseModel):
    route: Optional[str] = None

class ComparisonRequest(BaseModel):
    episodes: int = 10

class TraditionalTestRequest(BaseModel):
    strategy: str = 'rule_based'


# ═══════════════════════════════════════════════════════════════════════════════
# ROUTES — PAGES
# ═══════════════════════════════════════════════════════════════════════════════

@app.get('/', response_class=HTMLResponse)
async def landing(request: Request):
    return templates.TemplateResponse("landing.html", {"request": request})

@app.get('/control', response_class=HTMLResponse)
async def control(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get('/api/evaluation_log')
async def evaluation_log():
    try:
        with open('results/evaluation_log.txt', 'r', encoding='utf-8') as f:
            return HTMLResponse(content=f.read(), status_code=200)
    except Exception as e:
        return JSONResponse({'error': str(e)}, status_code=500)

@app.get('/results/{filename:path}')
async def serve_results(filename: str):
    return FileResponse(f"results/{filename}")


# ═══════════════════════════════════════════════════════════════════════════════
# ROUTES — STATE / ENVIRONMENT
# ═══════════════════════════════════════════════════════════════════════════════

@app.get('/api/state')
async def get_state():
    if not rl_system_loaded or sim_state is None:
        return JSONResponse({'error': 'RL system not loaded'}, status_code=500)
    return JSONResponse(sim_state.get_state_dict())

@app.get('/api/routes')
async def get_routes():
    if not rl_system_loaded or rl_env is None:
        return JSONResponse({'error': 'RL system not loaded'}, status_code=500)
    return JSONResponse({
        'routes':        rl_env.routes,
        'current_route': rl_env.route,
    })

@app.post('/api/change_route')
async def change_route(req: ChangeRouteRequest):
    if not rl_system_loaded or rl_env is None:
        return JSONResponse({'error': 'RL system not loaded'}, status_code=500)

    if req.route not in rl_env.routes:
        return JSONResponse({'error': f'Invalid route: {req.route}'}, status_code=400)

    rl_env.fixed_route = req.route
    sim_state.reset()

    return JSONResponse({'success': True, 'route': req.route, 'message': f'Switched to route: {req.route}'})

@app.post('/api/action')
async def take_action(req: ActionRequest):
    if not rl_system_loaded or sim_state is None:
        return JSONResponse({'error': 'RL system not loaded'}, status_code=500)

    if not (0 <= req.action < 9):
        return JSONResponse({'error': 'Invalid action'}, status_code=400)

    try:
        next_state, reward, done, info = sim_state.step(req.action)

        action_names = {
            0: 'E↓10% B↓10%', 1: 'E↓10% B→',   2: 'E↓10% B↑10%',
            3: 'E→ B↓10%',    4: 'E→ B→',      5: 'E→ B↑10%',
            6: 'E↑10% B↓10%', 7: 'E↑10% B→',   8: 'E↑10% B↑10%',
        }

        return JSONResponse({
            'success':        True,
            'action_name':    action_names[req.action],
            'econ_bookings':  int(info['econ_bookings']),
            'bus_bookings':   int(info['bus_bookings']),
            'total_bookings': int(info['econ_bookings'] + info['bus_bookings']),
            'econ_revenue':   float(info['econ_bookings'] * sim_state.env.econ_price),
            'bus_revenue':    float(info['bus_bookings']  * sim_state.env.bus_price),
            'total_revenue':  float(info['revenue']),
            'reward':         float(reward),
            'new_econ_price': float(info['econ_price']),
            'new_bus_price':  float(info['bus_price']),
            'done':           bool(done),
            'message':        f"Action: {action_names[req.action]} | Sold {info['econ_bookings']}E + {info['bus_bookings']}B | Reward: {reward:.1f}",
        })
    except Exception as e:
        import traceback; traceback.print_exc()
        return JSONResponse({'error': str(e)}, status_code=500)

@app.post('/api/disruption')
async def trigger_disruption(req: DisruptionRequest):
    if not rl_system_loaded or sim_state is None:
        return JSONResponse({'error': 'RL system not loaded'}, status_code=500)

    disruption_type = req.type
    sim_state.env.current_disruption = disruption_type
    
    if disruption_type != 'none':
        sim_state.env.disruption_duration = int(np.random.randint(1, 4))
    else:
        sim_state.env.disruption_duration = 0

    messages = {
        'weather':           '⛈️ Weather delay! Demand -40%',
        'pilot_strike':      '✊ Pilot strike! Demand -70%',
        'competitor_cancel': '✈️ Competitor cancelled! Demand +50%',
        'none':              '✅ Normal operations',
    }
    return JSONResponse({'success': True, 'disruption': disruption_type, 'message': messages.get(disruption_type, 'Unknown')})


@app.post('/api/reset')
async def reset_simulation(req: ResetRequest):
    if not rl_system_loaded or sim_state is None:
        return JSONResponse({'error': 'RL system not loaded'}, status_code=500)
    try:
        new_route = req.route
        if new_route and new_route in rl_env.routes:
            rl_env.fixed_route = new_route
        sim_state.reset()
        return JSONResponse({'success': True, 'message': 'RL environment reset', 'route': sim_state.env.route, 'calibrated': True})
    except Exception as e:
        return JSONResponse({'error': str(e)}, status_code=500)

@app.get('/api/history')
async def get_history():
    if not rl_system_loaded or sim_state is None:
        return JSONResponse({'error': 'RL system not loaded'}, status_code=500)
    history = sim_state.env.episode_history[-100:]
    return JSONResponse({'history': history})


# ═══════════════════════════════════════════════════════════════════════════════
# ROUTES — AGENT
# ═══════════════════════════════════════════════════════════════════════════════

@app.get('/api/agent_info')
async def get_agent_info():
    if not rl_system_loaded:
        return JSONResponse({'error': 'RL system not loaded'}, status_code=500)

    info = {
        'agent_loaded':    agent_loaded,
        'agent_status':    'trained' if agent_loaded else 'untrained',
        'state_size':      AGENT_CONFIG.get('state_size', 'unknown'),
        'action_size':     9,
        'epsilon':         float(rl_agent.epsilon)         if rl_agent else 0.0,
        'device':          str(rl_agent.device)            if rl_agent else 'unknown',
        'training_steps':  rl_agent.training_steps         if rl_agent else 0,
        'episodes_trained': rl_agent.episode_count         if rl_agent else 0,
    }
    return JSONResponse(info)


@app.get('/api/ai_recommendation')
async def get_ai_recommendation():
    """RL agent's recommended action with context-aware reasoning."""
    global _rec_cache, _rec_cache_time

    if not rl_system_loaded or sim_state is None:
        return JSONResponse({'error': 'RL system not loaded'}, status_code=500)
    if rl_agent is None:
        return JSONResponse({'error': 'RL agent not initialized'}, status_code=500)

    if time.time() - _rec_cache_time < 3 and _rec_cache:
        return JSONResponse(_rec_cache)

    try:
        state = sim_state.current_state
        if not isinstance(state, np.ndarray):
            state = np.array(state, dtype=np.float32)

        env = sim_state.env

        q_values    = rl_agent.get_action_distribution(state)
        action      = int(np.argmax(q_values))
        q_value     = float(q_values[action])

        action_names = {
            0: 'E↓10% B↓10%', 1: 'E↓10% B→',   2: 'E↓10% B↑10%',
            3: 'E→ B↓10%',    4: 'E→ B→',        5: 'E→ B↑10%',
            6: 'E↑10% B↓10%', 7: 'E↑10% B→',    8: 'E↑10% B↑10%',
        }
        action_name = action_names.get(action, f"Action {action}")

        q_max  = float(np.max(q_values))
        q_mean = float(np.mean(q_values))
        q_std  = float(np.std(q_values))
        confidence = (min(0.97, 0.5 + (q_max - q_mean) / (2 * q_std + 1e-8)) if q_std > 0.5 else 0.40)

        econ_load   = env.econ_sold / env.econ_seats_total
        bus_load    = env.bus_sold  / env.bus_seats_total
        total_load  = (env.econ_sold + env.bus_sold) / env.total_seats
        days_left   = env.days_to_departure

        econ_comp_avg = np.mean(list(env.econ_competitors.values())) if env.econ_competitors else env.econ_price
        bus_comp_avg  = np.mean(list(env.bus_competitors.values()))  if env.bus_competitors  else env.bus_price
        econ_ratio    = env.econ_price / econ_comp_avg if econ_comp_avg > 0 else 1.0
        bus_ratio     = env.bus_price  / bus_comp_avg  if bus_comp_avg  > 0 else 1.0

        if q_std < 0.5 and not agent_loaded:
            if days_left < 7 and total_load < 0.6:
                action = 0; action_name = action_names[0]
                reason = f"⚠️ UNTRAINED — Rule: {days_left}d left, only {total_load*100:.0f}% full → Drop prices!"
            elif econ_ratio > 1.15 and econ_load < 0.7:
                action = 1; action_name = action_names[1]
                reason = f"⚠️ UNTRAINED — Rule: Economy overpriced {(econ_ratio-1)*100:.0f}% → Lower Economy"
            elif bus_ratio > 1.20 and bus_load < 0.6:
                action = 3; action_name = action_names[3]
                reason = f"⚠️ UNTRAINED — Rule: Business overpriced {(bus_ratio-1)*100:.0f}% → Lower Business"
            elif econ_ratio < 0.90 and econ_load > 0.75:
                action = 8; action_name = action_names[8]
                reason = "⚠️ UNTRAINED — Rule: Underpriced with high demand → Raise prices"
            elif env.current_disruption == 'competitor_cancel':
                action = 8; action_name = action_names[8]
                reason = "⚠️ UNTRAINED — Rule: Competitor cancelled → Raise prices!"
            elif env.current_disruption in ['weather', 'pilot_strike']:
                action = 0; action_name = action_names[0]
                reason = f"⚠️ UNTRAINED — Rule: {env.current_disruption} → Lower prices"
            else:
                action = 4; action_name = action_names[4]
                reason = "⚠️ UNTRAINED agent — Train model for better recommendations."
            confidence = 0.5

        elif agent_loaded:
            reasons = []
            if days_left < 7:          reasons.append(f"⏰ Only {days_left}d left")
            if total_load < 0.5:       reasons.append(f"🪑 Low fill: {total_load*100:.0f}%")
            if total_load > 0.85:      reasons.append(f"🔥 High demand: {total_load*100:.0f}%")
            if econ_ratio > 1.10:      reasons.append(f"📉 Econ overpriced {(econ_ratio-1)*100:.0f}% vs mkt")
            if econ_ratio < 0.92:      reasons.append(f"📈 Econ underpriced {(1-econ_ratio)*100:.0f}% vs mkt")
            if bus_ratio > 1.15:       reasons.append(f"📉 Biz overpriced {(bus_ratio-1)*100:.0f}% vs mkt")
            if bus_ratio < 0.90:       reasons.append(f"📈 Biz underpriced {(1-bus_ratio)*100:.0f}% vs mkt")
            if env.current_disruption != 'none':
                reasons.append(f"⚠️ {env.current_disruption}")
            ctx    = " | ".join(reasons) if reasons else "Stable market conditions"
            reason = f"🤖 RL Agent → {action_name} | {ctx} | Q-spread: {q_std:.2f}"
        else:
            reason = f"⚠️ Model uncertain (Q-spread: {q_std:.2f}) — Train more episodes"

        q_shifted     = q_values - np.max(q_values)
        softmax_probs = np.exp(q_shifted) / np.sum(np.exp(q_shifted))
        top3_indices  = np.argsort(q_values)[::-1][:3]
        top3_actions  = [
            {
                'action':      int(i),
                'name':        action_names[int(i)],
                'q_value':     float(q_values[i]),
                'probability': float(softmax_probs[i]),
            }
            for i in top3_indices
        ]

        result = {
            'action':       int(action),
            'action_name':  action_name,
            'reason':       reason,
            'confidence':   float(confidence),
            'q_value':      float(q_value),
            'q_spread':     float(q_std),
            'top3_actions': top3_actions,
            'agent_status': 'trained' if agent_loaded else 'untrained',
            'market_context': {
                'econ_price':     float(env.econ_price),
                'bus_price':      float(env.bus_price),
                'econ_vs_market': f"{((econ_ratio - 1) * 100):+.1f}%",
                'bus_vs_market':  f"{((bus_ratio  - 1) * 100):+.1f}%",
                'econ_load':      float(econ_load * 100),
                'bus_load':       float(bus_load  * 100),
                'days_left':      int(days_left),
                'disruption':     env.current_disruption,
            },
        }

        _rec_cache      = result
        _rec_cache_time = time.time()
        return JSONResponse(result)

    except Exception as e:
        import traceback; traceback.print_exc()
        return JSONResponse({'error': str(e)}, status_code=500)


# ═══════════════════════════════════════════════════════════════════════════════
# ROUTES — COMPARISON
# ═══════════════════════════════════════════════════════════════════════════════

@app.post('/api/run_comparison')
async def run_comparison(req: ComparisonRequest):
    global comparison_results

    if not rl_system_loaded or rl_env is None:
        return JSONResponse({'error': 'RL system not loaded'}, status_code=500)

    try:
        num_episodes = req.episodes
        print(f"\n🔄 Running comparison ({num_episodes} episodes per strategy)…")

        if not hasattr(rl_env, '_route_stats_path'):
            rl_env._route_stats_path = CALIBRATION_PATH

        comparison_results = compare_all_strategies(
            env=rl_env,
            rl_agent=rl_agent if agent_loaded else None,
            num_episodes=num_episodes,
            verbose=True,
        )

        formatted_results = {}
        for strategy_name, metrics in comparison_results.items():
            formatted_results[strategy_name] = {
                'name':            strategy_name.replace('_', ' ').title(),
                'avg_revenue':     float(metrics['avg_revenue']),
                'std_revenue':     float(metrics['std_revenue']),
                'min_revenue':     float(metrics.get('min_revenue', metrics['avg_revenue'])),
                'max_revenue':     float(metrics.get('max_revenue', metrics['avg_revenue'])),
                'avg_load_factor': float(metrics['avg_load_factor'] * 100),
                'avg_econ_load':   float(metrics['avg_econ_load']   * 100),
                'avg_bus_load':    float(metrics['avg_bus_load']    * 100),
                'revenues':        [float(r)  for r  in metrics['revenues']],
                'load_factors':    [float(lf * 100) for lf in metrics['load_factors']],
            }

        if 'rl_agent' in formatted_results and agent_loaded:
            rl_revenue   = formatted_results['rl_agent']['avg_revenue']
            trad_names   = [k for k in formatted_results if k != 'rl_agent']
            best_name    = max(trad_names, key=lambda k: formatted_results[k]['avg_revenue'])
            best_revenue = formatted_results[best_name]['avg_revenue']
            improvement  = (rl_revenue - best_revenue) / best_revenue * 100

            formatted_results['comparison_summary'] = {
                'rl_revenue':               rl_revenue,
                'best_traditional':         best_name,
                'best_traditional_revenue': best_revenue,
                'improvement_percent':      float(improvement),
                'rl_advantage':             rl_revenue > best_revenue,
            }

        return JSONResponse({
            'success':      True,
            'results':      formatted_results,
            'num_episodes': num_episodes,
            'message':      f'Comparison complete: {len(comparison_results)} strategies evaluated',
        })
    except Exception as e:
        import traceback; traceback.print_exc()
        return JSONResponse({'error': str(e)}, status_code=500)

@app.get('/api/get_comparison')
async def get_comparison():
    if comparison_results is None:
        return JSONResponse({'error': 'No comparison results. Run comparison first.'}, status_code=404)

    formatted_results = {}
    for strategy_name, metrics in comparison_results.items():
        formatted_results[strategy_name] = {
            'name':            strategy_name.replace('_', ' ').title(),
            'avg_revenue':     float(metrics['avg_revenue']),
            'std_revenue':     float(metrics['std_revenue']),
            'min_revenue':     float(metrics.get('min_revenue', metrics['avg_revenue'])),
            'max_revenue':     float(metrics.get('max_revenue', metrics['avg_revenue'])),
            'avg_load_factor': float(metrics['avg_load_factor'] * 100),
            'avg_econ_load':   float(metrics['avg_econ_load']   * 100),
            'avg_bus_load':    float(metrics['avg_bus_load']    * 100),
            'revenues':        [float(r)  for r  in metrics['revenues']],
            'load_factors':    [float(lf * 100) for lf in metrics['load_factors']],
        }

    if 'rl_agent' in formatted_results and agent_loaded:
        rl_revenue   = formatted_results['rl_agent']['avg_revenue']
        trad_names   = [k for k in formatted_results if k != 'rl_agent']
        best_name    = max(trad_names, key=lambda k: formatted_results[k]['avg_revenue'])
        best_revenue = formatted_results[best_name]['avg_revenue']
        improvement  = (rl_revenue - best_revenue) / best_revenue * 100

        formatted_results['comparison_summary'] = {
            'rl_revenue':               rl_revenue,
            'best_traditional':         best_name,
            'best_traditional_revenue': best_revenue,
            'improvement_percent':      float(improvement),
            'rl_advantage':             rl_revenue > best_revenue,
        }

    return JSONResponse({'success': True, 'results': formatted_results})


@app.post('/api/test_traditional')
async def test_traditional(req: TraditionalTestRequest):
    if not rl_system_loaded or rl_env is None:
        return JSONResponse({'error': 'RL system not loaded'}, status_code=500)

    strategy_name = req.strategy

    if strategy_name not in TRADITIONAL_STRATEGIES:
        return JSONResponse({'error': f'Unknown strategy: {strategy_name}'}, status_code=400)

    try:
        strategy_fn = TRADITIONAL_STRATEGIES[strategy_name]
        test_env = AirlineRevenueEnv(
            route_stats_path=CALIBRATION_PATH,
            fixed_route=rl_env.fixed_route,
        )
        state, _ = test_env.reset()
        done         = False
        total_reward = 0
        actions_taken = []

        while not done:
            action = int(strategy_fn(test_env))
            state, reward, terminated, truncated, info = test_env.step(action)
            done          = terminated or truncated
            total_reward += reward
            actions_taken.append(action)

        summary = test_env.get_episode_summary()

        return JSONResponse({
            'success':       True,
            'strategy':      strategy_name.replace('_', ' ').title(),
            'total_revenue': float(summary['total_revenue']),
            'load_factor':   float(summary['load_factor']       * 100),
            'econ_load':     float(summary['econ_load_factor']   * 100),
            'bus_load':      float(summary['bus_load_factor']    * 100),
            'total_reward':  float(total_reward),
            'actions_taken': len(actions_taken),
            'message':       f"{strategy_name.replace('_', ' ').title()} completed: ₹{summary['total_revenue']:,.0f} revenue",
        })

    except Exception as e:
        import traceback; traceback.print_exc()
        return JSONResponse({'error': str(e)}, status_code=500)

if __name__ == '__main__':
    import uvicorn
    print("\n" + "=" * 80)
    print("  🚀 RL-INTEGRATED MULTI-CLASS AIRLINE DASHBOARD")
    print("=" * 80)

    port = FLASK_CONFIG.get('port', 8080)
    host = FLASK_CONFIG.get('host', '0.0.0.0')
    print(f"\n   Open in browser: http://127.0.0.1:{port}\n")
    uvicorn.run("app:app", host=host, port=port, reload=False)
