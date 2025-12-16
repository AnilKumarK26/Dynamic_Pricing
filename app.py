from flask import Flask, render_template, jsonify, request, session
import numpy as np
import pandas as pd
import json
from datetime import datetime
import os
import pickle

app = Flask(__name__)
app.secret_key = 'airline_rl_secret_key_2024'

# Global variables to store simulation state
class SimulationState:
    def __init__(self):
        self.calibrated = False
        self.base_price = 6000
        self.price_min = 3000
        self.price_max = 15000
        self.competitor_prices = {
            'SpiceJet': 5950,
            'AirAsia': 5960,
            'Vistara': 5940,
            'IndiGo': 5970
        }
        self.reset()
    
    def load_calibration(self, route_stats):
        """Load REAL calibrated data from analyze_data.py"""
        try:
            # Get REAL prices from YOUR data
            price_stats = route_stats['price_stats']
            self.base_price = price_stats['mean']
            self.price_min = price_stats['q25']
            self.price_max = price_stats['q75'] * 1.5
            
            # Get REAL competitor prices from YOUR data
            self.competitor_prices = route_stats['competitor_prices'].copy()
            self._original_competitor_prices = self.competitor_prices.copy()
            
            self.calibrated = True
            print(f"✓ Calibration loaded!")
            print(f"  Base Price: ₹{self.base_price:.0f}")
            print(f"  Price Range: ₹{self.price_min:.0f} - ₹{self.price_max:.0f}")
            print(f"  Competitors: {list(self.competitor_prices.keys())}")
            
            return True
        except Exception as e:
            print(f"⚠️  Error loading calibration: {e}")
            self.calibrated = False
            return False
    
    def reset(self):
        """Reset simulation state"""
        self.current_step = 0
        self.current_price = self.base_price
        self.seats_sold = 0
        self.total_seats = 180
        self.total_revenue = 0
        self.days_to_departure = 90
        self.disruption = 'none'
        self.history = []
        
        # Reset competitor prices to original calibrated values
        if hasattr(self, '_original_competitor_prices'):
            self.competitor_prices = self._original_competitor_prices.copy()

sim_state = SimulationState()

# Load REAL calibration data
def load_calibration():
    """Load calibration from analyze_data.py output"""
    calibration_path = 'data/route_stats.pkl'
    
    if os.path.exists(calibration_path):
        print(f"\n✓ Found calibration file: {calibration_path}")
        try:
            with open(calibration_path, 'rb') as f:
                route_stats = pickle.load(f)
            
            # Use the first available route
            if route_stats:
                first_route = list(route_stats.keys())[0]
                print(f"   Using route: {first_route}")
                if sim_state.load_calibration(route_stats[first_route]):
                    return True
            return False
        except Exception as e:
            print(f"⚠️  Error loading calibration: {e}")
            return False
    else:
        print(f"\n⚠️  No calibration file at {calibration_path}")
        print(f"   Run: python analyze_data.py")
        return False

# Load flight data
def load_data():
    """Load REAL flight dataset"""
    data_path = 'data/flight_data.csv'
    if os.path.exists(data_path):
        df = pd.read_csv(data_path)
        print(f"✓ Loaded {len(df):,} real flight records")
        return df
    else:
        print("⚠️  No flight_data.csv found!")
        return None

flight_data = load_data()
calibration_loaded = load_calibration()

@app.route('/')
def index():
    """Main dashboard page"""
    return render_template('index.html')

@app.route('/api/state')
def get_state():
    """Get current simulation state"""
    load_factor = (sim_state.seats_sold / sim_state.total_seats) * 100
    
    return jsonify({
        'current_price': round(sim_state.current_price, 2),
        'seats_sold': sim_state.seats_sold,
        'total_seats': sim_state.total_seats,
        'load_factor': round(load_factor, 1),
        'total_revenue': round(sim_state.total_revenue, 2),
        'days_to_departure': sim_state.days_to_departure,
        'disruption': sim_state.disruption,
        'competitor_prices': {k: round(v, 2) for k, v in sim_state.competitor_prices.items()},
        'step': sim_state.current_step,
        'calibrated': sim_state.calibrated
    })

@app.route('/api/action', methods=['POST'])
def take_action():
    """Execute pricing action"""
    data = request.json
    action = data.get('action', 'hold')
    
    # Price adjustments
    price_changes = {
        'decrease_20': -0.20,
        'decrease_10': -0.10,
        'hold': 0.0,
        'increase_10': 0.10,
        'increase_20': 0.20
    }
    
    adjustment = price_changes.get(action, 0.0)
    old_price = sim_state.current_price
    sim_state.current_price *= (1 + adjustment)
    
    # Keep price within calibrated range
    sim_state.current_price = np.clip(
        sim_state.current_price,
        sim_state.price_min,
        sim_state.price_max
    )
    
    # Calculate REALISTIC demand based on price competitiveness
    avg_competitor = np.mean(list(sim_state.competitor_prices.values()))
    price_ratio = sim_state.current_price / avg_competitor
    
    # Exponential price sensitivity (realistic consumer behavior)
    price_elasticity = 2.0
    price_factor = np.exp(-price_elasticity * (price_ratio - 1))
    
    # Time factor (more bookings as departure approaches)
    time_factor = 0.3 + (1 - sim_state.days_to_departure / 90) * 1.2
    
    # Disruption effects
    disruption_factor = 1.0
    if sim_state.disruption == 'weather':
        disruption_factor = 0.6
    elif sim_state.disruption == 'pilot_strike':
        disruption_factor = 0.3
    elif sim_state.disruption == 'competitor_cancel':
        disruption_factor = 1.5
    
    # Calculate expected bookings
    base_demand_rate = 0.15
    demand = base_demand_rate * time_factor * price_factor * disruption_factor
    expected_bookings = demand * sim_state.total_seats * 0.15
    
    # Add randomness with Poisson
    bookings = np.random.poisson(expected_bookings)
    bookings = min(bookings, sim_state.total_seats - sim_state.seats_sold)
    bookings = max(0, bookings)
    
    # Update state
    sim_state.seats_sold += bookings
    revenue_this_step = bookings * sim_state.current_price
    sim_state.total_revenue += revenue_this_step
    sim_state.days_to_departure -= 1
    sim_state.current_step += 1
    
    # Update competitor prices (realistic random walk)
    price_volatility = (sim_state.price_max - sim_state.price_min) * 0.01
    for airline in sim_state.competitor_prices:
        change = np.random.normal(0, price_volatility)
        sim_state.competitor_prices[airline] += change
        sim_state.competitor_prices[airline] = np.clip(
            sim_state.competitor_prices[airline],
            sim_state.price_min,
            sim_state.price_max
        )
    
    # Record history
    sim_state.history.append({
        'step': sim_state.current_step,
        'price': round(sim_state.current_price, 2),
        'bookings': bookings,
        'revenue': round(revenue_this_step, 2),
        'seats_sold': sim_state.seats_sold,
        'disruption': sim_state.disruption,
        'demand_factor': round(demand, 3)
    })
    
    # Calculate reward
    reward = revenue_this_step / 1000
    if sim_state.days_to_departure < 7 and sim_state.seats_sold < sim_state.total_seats * 0.5:
        reward -= 5
    
    done = (sim_state.days_to_departure <= 0) or (sim_state.seats_sold >= sim_state.total_seats)
    
    return jsonify({
        'success': True,
        'bookings': bookings,
        'revenue': round(revenue_this_step, 2),
        'reward': round(reward, 2),
        'new_price': round(sim_state.current_price, 2),
        'old_price': round(old_price, 2),
        'done': done,
        'demand_factor': round(demand, 3),
        'message': f'Sold {bookings} seats at ₹{sim_state.current_price:.0f} (demand: {demand*100:.1f}%)'
    })

@app.route('/api/disruption', methods=['POST'])
def trigger_disruption():
    """Trigger a disruption event"""
    data = request.json
    disruption_type = data.get('type', 'none')
    
    sim_state.disruption = disruption_type
    
    messages = {
        'weather': '⛈️ Weather delay triggered! Demand decreased by 40%.',
        'pilot_strike': '✊ Pilot strike! Demand decreased by 70%.',
        'competitor_cancel': '✈️ Competitor cancelled flights! Demand increased by 50%!',
        'none': '✅ Normal operations resumed.'
    }
    
    print(f"Disruption: {disruption_type}")
    
    return jsonify({
        'success': True,
        'disruption': disruption_type,
        'message': messages.get(disruption_type, 'Unknown disruption')
    })

@app.route('/api/reset', methods=['POST'])
def reset_simulation():
    """Reset simulation to initial state"""
    try:
        sim_state.reset()
        
        print(f"\n🔄 Simulation reset")
        print(f"   Base Price: ₹{sim_state.base_price:.0f}")
        print(f"   Seats: {sim_state.seats_sold}/{sim_state.total_seats}")
        
        return jsonify({
            'success': True,
            'message': 'Simulation reset successfully',
            'calibrated': sim_state.calibrated,
            'base_price': round(sim_state.base_price, 2)
        })
    except Exception as e:
        print(f"❌ Reset failed: {e}")
        return jsonify({
            'success': False,
            'message': f'Reset failed: {str(e)}'
        }), 500

@app.route('/api/history')
def get_history():
    """Get simulation history"""
    return jsonify({
        'history': sim_state.history[-100:]
    })

@app.route('/api/ai_recommendation')
def get_ai_recommendation():
    """Get AI agent's recommended action with DYNAMIC confidence"""
    
    avg_competitor = np.mean(list(sim_state.competitor_prices.values()))
    price_diff = sim_state.current_price - avg_competitor
    load_factor = sim_state.seats_sold / sim_state.total_seats
    days_left = sim_state.days_to_departure
    
    # Calculate price position
    price_ratio = sim_state.current_price / avg_competitor
    
    # Get cheapest competitor
    min_competitor_price = min(sim_state.competitor_prices.values())
    max_competitor_price = max(sim_state.competitor_prices.values())
    cheapest_airline = min(sim_state.competitor_prices, key=sim_state.competitor_prices.get)
    
    # REALISTIC AI Decision Logic with VARYING confidence
    
    # Critical: Near departure with low bookings
    if days_left < 7:
        if load_factor < 0.4:
            recommendation = 'decrease_20'
            reason = f'⚠️ URGENT: Only {days_left} days left with {load_factor*100:.0f}% full - aggressive pricing needed!'
            confidence = 0.96
        elif load_factor < 0.6:
            recommendation = 'decrease_10'
            reason = f'⏰ {days_left} days left, {load_factor*100:.0f}% full - discount to fill seats'
            confidence = 0.89
        elif load_factor < 0.8:
            recommendation = 'hold'
            reason = f'✓ Good progress at {load_factor*100:.0f}% with {days_left} days left'
            confidence = 0.81
        else:
            recommendation = 'hold'
            reason = f'🎯 Excellent load factor ({load_factor*100:.0f}%) - maintain price'
            confidence = 0.93
    
    # We're significantly more expensive
    elif price_ratio > 1.20:
        if load_factor < 0.2:
            recommendation = 'decrease_20'
            reason = f'📉 Price {(price_ratio-1)*100:.0f}% above market - losing to competition'
            confidence = 0.94
        else:
            recommendation = 'decrease_10'
            reason = f'💰 Price {(price_ratio-1)*100:.0f}% above average - becoming uncompetitive'
            confidence = 0.88
    
    # Moderately expensive
    elif price_ratio > 1.10:
        if load_factor < 0.3:
            recommendation = 'decrease_10'
            reason = f'📊 Above market price with low bookings - adjust down'
            confidence = 0.82
        else:
            recommendation = 'hold'
            reason = f'✓ Slightly premium but maintaining demand'
            confidence = 0.75
    
    # We're significantly cheaper
    elif price_ratio < 0.85:
        if load_factor > 0.7:
            recommendation = 'increase_20'
            reason = f'💎 Strong demand at lowest price - capture more revenue!'
            confidence = 0.92
        elif load_factor > 0.5:
            recommendation = 'increase_10'
            reason = f'📈 Good bookings, {(1-price_ratio)*100:.0f}% below market - room to grow'
            confidence = 0.87
        else:
            recommendation = 'increase_10'
            reason = f'💵 Underpriced at ₹{sim_state.current_price:.0f} vs ₹{avg_competitor:.0f} avg'
            confidence = 0.79
    
    # Moderately cheap
    elif price_ratio < 0.95:
        if load_factor > 0.6:
            recommendation = 'increase_10'
            reason = f'✨ Competitive price with solid bookings - optimize revenue'
            confidence = 0.84
        else:
            recommendation = 'hold'
            reason = f'⚖️ Good price position, monitor demand'
            confidence = 0.73
    
    # Disruption-based decisions
    elif sim_state.disruption == 'competitor_cancel':
        recommendation = 'increase_20'
        reason = f'🚨 Competitor issues - surge pricing opportunity!'
        confidence = 0.95
    
    elif sim_state.disruption == 'pilot_strike':
        recommendation = 'decrease_10'
        reason = f'✊ Our strike affecting demand - stay competitive'
        confidence = 0.86
    
    elif sim_state.disruption == 'weather':
        recommendation = 'hold'
        reason = f'⛈️ Weather disruption - maintain steady pricing'
        confidence = 0.77
    
    # Early booking surge
    elif load_factor > 0.8 and days_left > 30:
        recommendation = 'increase_10'
        reason = f'🔥 Strong early demand ({load_factor*100:.0f}% at D-{days_left}) - maximize revenue'
        confidence = 0.91
    
    # Perfectly positioned
    elif abs(price_ratio - 1.0) < 0.05:
        if load_factor > 0.5:
            recommendation = 'hold'
            reason = f'🎯 Perfect position: market price with {load_factor*100:.0f}% bookings'
            confidence = 0.88
        else:
            recommendation = 'decrease_10'
            reason = f'⚠️ At market price but low bookings - slight discount'
            confidence = 0.72
    
    # Slightly below market
    elif price_ratio < 1.0:
        recommendation = 'hold'
        reason = f'✓ Good competitive position at ₹{sim_state.current_price:.0f}'
        confidence = 0.76
    
    # Default: slightly above market
    else:
        if load_factor < 0.4:
            recommendation = 'decrease_10'
            reason = f'📉 Above market with weak demand - adjust'
            confidence = 0.74
        else:
            recommendation = 'hold'
            reason = f'➡️ Monitor situation, consider small adjustment'
            confidence = 0.68
    
    # Market context
    market_context = {
        'our_price': round(sim_state.current_price, 0),
        'market_avg': round(avg_competitor, 0),
        'cheapest_competitor': f"{cheapest_airline} (₹{min_competitor_price:.0f})",
        'price_vs_market': f"{((price_ratio - 1) * 100):+.1f}%",
        'load_factor': round(load_factor * 100, 1),
        'days_to_departure': days_left,
        'disruption': sim_state.disruption
    }
    
    return jsonify({
        'action': recommendation,
        'reason': reason,
        'confidence': round(confidence, 2),
        'market_context': market_context
    })

@app.route('/api/analytics')
def get_analytics():
    """Get analytics data"""
    if not sim_state.history:
        return jsonify({'error': 'No data available'})
    
    df = pd.DataFrame(sim_state.history)
    
    analytics = {
        'total_revenue': round(sim_state.total_revenue, 2),
        'avg_price': round(df['price'].mean(), 2),
        'total_bookings': int(df['bookings'].sum()),
        'load_factor': round((sim_state.seats_sold / sim_state.total_seats) * 100, 1),
        'revenue_per_seat': round(sim_state.total_revenue / max(sim_state.seats_sold, 1), 2),
        'steps_completed': sim_state.current_step,
        'avg_demand': round(df['demand_factor'].mean() * 100, 1) if 'demand_factor' in df else 0
    }
    
    return jsonify(analytics)

@app.route('/api/dataset_info')
def get_dataset_info():
    """Get information about the dataset"""
    if flight_data is None:
        return jsonify({'error': 'No flight data loaded'})
    
    # Get available routes
    routes = {}
    if 'route' in flight_data.columns:
        route_counts = flight_data['route'].value_counts()
        for route, count in route_counts.head(10).items():
            route_data = flight_data[flight_data['route'] == route]
            routes[route] = {
                'count': int(count),
                'avg_price': float(route_data['price'].mean()),
                'airlines': int(route_data['airline'].nunique())
            }
    
    info = {
        'total_flights': len(flight_data),
        'airlines': flight_data['airline'].nunique() if 'airline' in flight_data.columns else 0,
        'routes': routes,
        'avg_price': round(flight_data['price'].mean(), 2) if 'price' in flight_data.columns else 0,
        'price_range': {
            'min': round(flight_data['price'].min(), 2) if 'price' in flight_data.columns else 0,
            'max': round(flight_data['price'].max(), 2) if 'price' in flight_data.columns else 0
        },
        'calibrated': sim_state.calibrated
    }
    return jsonify(info)

if __name__ == '__main__':
    os.makedirs('templates', exist_ok=True)
    os.makedirs('static/css', exist_ok=True)
    os.makedirs('static/js', exist_ok=True)
    
    print("\n" + "="*80)
    print("  🚀 AIRLINE RL DASHBOARD STARTING")
    print("="*80)
    
    if sim_state.calibrated:
        print("\n  ✅ Using CALIBRATED environment with YOUR real data")
        print(f"  💰 Base Price: ₹{sim_state.base_price:.0f}")
        print(f"  📉 Price Range: ₹{sim_state.price_min:.0f} - ₹{sim_state.price_max:.0f}")
        print(f"  🏢 Competitors: {len(sim_state.competitor_prices)}")
        for airline, price in sim_state.competitor_prices.items():
            print(f"     - {airline}: ₹{price:.0f}")
    else:
        print("\n  ⚠️  Using DEFAULT values (not calibrated)")
        print("  📋 Run: python analyze_data.py (to calibrate with YOUR data)")
        print(f"  💰 Default Base Price: ₹{sim_state.base_price:.0f}")
    
    print("\n  📊 Dashboard: http://localhost:5000")
    print("  📁 API Endpoints:")
    print("     - POST /api/reset  ✅ FIXED")
    print("     - GET  /api/ai_recommendation  ✅ DYNAMIC CONFIDENCE (68%-96%)")
    print("\n  Press Ctrl+C to stop\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)