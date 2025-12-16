"""
Calibrated Airline Revenue Management Environment
Uses REAL flight data statistics for realistic simulation
File: environment/airline_env.py
"""

import numpy as np
import pandas as pd
import gym
from gym import spaces
import random
import pickle
import os


class AirlineRevenueEnv(gym.Env):
    """
    RL Environment for Airline Dynamic Pricing
    Calibrated using REAL flight data for specific routes
    """
    
    def __init__(self, route='Delhi-Mumbai', route_stats=None, flight_data=None):
        super(AirlineRevenueEnv, self).__init__()
        
        self.route = route
        print(f"\n🎯 Initializing Environment for Route: {route}")
        
        # Load calibrated parameters from REAL data
        if route_stats is not None:
            self._load_from_stats(route_stats)
        elif flight_data is not None:
            self._calibrate_from_data(flight_data)
        else:
            print("⚠️  No calibration data provided - using default parameters")
            self._load_defaults()
        
        # Action space: 5 discrete pricing actions
        # 0: -20%, 1: -10%, 2: 0%, 3: +10%, 4: +20%
        self.action_space = spaces.Discrete(5)
        
        # State space: [current_price, competitor_avg, days_to_dept, 
        #                seats_remaining_pct, demand_level, disruption, hour]
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0, 0, 0, 0, 0]),
            high=np.array([50000, 50000, 90, 1, 1, 1, 23]),
            dtype=np.float32
        )
        
        # Episode parameters
        self.max_steps = self.max_days
        self.current_step = 0
        
        # State variables (will be initialized in reset)
        self.current_price = None
        self.seats_sold = 0
        self.days_to_departure = None
        self.total_revenue = 0
        self.disruption_active = False
        self.current_disruption = 'none'
        
        # Disruption types
        self.disruption_types = ['none', 'weather', 'pilot_strike', 'competitor_cancel']
        
        print(f"✓ Environment initialized")
        print(f"  Base Price: ₹{self.base_price:.0f}")
        print(f"  Price Range: ₹{self.price_min:.0f} - ₹{self.price_max:.0f}")
        print(f"  Total Seats: {self.total_seats}")
        print(f"  Competitors: {len(self.competitor_prices)}")
    
    def _load_from_stats(self, route_stats):
        """Load environment parameters from analyzed route statistics"""
        print("📊 Loading calibrated parameters from route analysis...")
        
        price_stats = route_stats['price_stats']
        
        # Pricing parameters from REAL data
        self.base_price = price_stats['mean']
        self.price_std = price_stats['std']
        self.price_min = price_stats['q25']
        self.price_max = price_stats['q75'] * 1.5
        
        # Competitor prices from REAL data
        self.competitor_prices = route_stats['competitor_prices'].copy()
        
        # Aircraft and timing
        self.total_seats = 180
        self.max_days = 90
        
        # Demand calibration (realistic booking patterns)
        self.base_demand_rate = 0.15  # 15% capacity per day average
        self.price_elasticity = 2.0
        
    def _calibrate_from_data(self, flight_data):
        """Calibrate environment from DataFrame"""
        print("📊 Calibrating from flight data...")
        
        route_df = flight_data[flight_data['route'] == self.route]
        
        if len(route_df) == 0:
            print(f"⚠️  No data for route {self.route}, using defaults")
            self._load_defaults()
            return
        
        # Calculate statistics
        self.base_price = route_df['price'].mean()
        self.price_std = route_df['price'].std()
        self.price_min = route_df['price'].quantile(0.25)
        self.price_max = route_df['price'].quantile(0.75) * 1.5
        
        # Get competitor prices
        self.competitor_prices = {}
        for airline in route_df['airline'].unique():
            avg_price = route_df[route_df['airline'] == airline]['price'].mean()
            self.competitor_prices[airline] = avg_price
        
        self.total_seats = 180
        self.max_days = 90
        self.base_demand_rate = 0.15
        self.price_elasticity = 2.0
        
        print(f"  ✓ Calibrated with {len(route_df)} flights")
    
    def _load_defaults(self):
        """Load default parameters if no data available"""
        self.base_price = 6000
        self.price_std = 500
        self.price_min = 4000
        self.price_max = 10000
        
        self.competitor_prices = {
            'Competitor_A': 5950,
            'Competitor_B': 5960,
            'Competitor_C': 5940,
        }
        
        self.total_seats = 180
        self.max_days = 90
        self.base_demand_rate = 0.15
        self.price_elasticity = 2.0
    
    def reset(self):
        """Reset environment to initial state"""
        self.current_step = 0
        self.current_price = self.base_price
        self.seats_sold = 0
        self.days_to_departure = self.max_days
        self.total_revenue = 0
        self.disruption_active = False
        self.current_disruption = 'none'
        
        return self._get_state()
    
    def _get_state(self):
        """Get current state observation"""
        competitor_avg = np.mean(list(self.competitor_prices.values()))
        seats_remaining_pct = (self.total_seats - self.seats_sold) / self.total_seats
        demand_level = self._calculate_demand()
        disruption_flag = 1.0 if self.disruption_active else 0.0
        current_hour = random.randint(0, 23)
        
        state = np.array([
            self.current_price,
            competitor_avg,
            self.days_to_departure,
            seats_remaining_pct,
            demand_level,
            disruption_flag,
            current_hour
        ], dtype=np.float32)
        
        return state
    
    def _calculate_demand(self):
        """
        Calculate demand based on REALISTIC factors
        Uses calibrated price elasticity and booking curves
        """
        # Time-based demand (booking curve)
        # More bookings closer to departure
        days_ratio = self.days_to_departure / self.max_days
        time_factor = 0.3 + (1 - days_ratio) * 1.2  # Increases as departure approaches
        
        # Price competitiveness
        competitor_avg = np.mean(list(self.competitor_prices.values()))
        price_ratio = self.current_price / competitor_avg
        
        # Exponential price sensitivity (realistic consumer behavior)
        price_factor = np.exp(-self.price_elasticity * (price_ratio - 1))
        
        # Disruption effects
        disruption_factor = 1.0
        if self.current_disruption == 'weather':
            disruption_factor = 0.6
        elif self.current_disruption == 'pilot_strike':
            disruption_factor = 0.3
        elif self.current_disruption == 'competitor_cancel':
            disruption_factor = 1.5
        
        # Combine all factors
        demand = self.base_demand_rate * time_factor * price_factor * disruption_factor
        
        return np.clip(demand, 0, 1)
    
    def _simulate_bookings(self, demand_level):
        """
        Simulate realistic bookings based on demand
        Uses Poisson distribution for natural randomness
        """
        seats_available = self.total_seats - self.seats_sold
        
        # Expected bookings per day
        expected_bookings = demand_level * self.total_seats * 0.15
        
        # Add realistic randomness with Poisson
        bookings = np.random.poisson(expected_bookings)
        
        # Can't sell more than available
        bookings = min(bookings, seats_available)
        
        return max(0, bookings)
    
    def _trigger_disruption(self):
        """Randomly trigger disruptions (5% chance)"""
        if random.random() < 0.05:
            self.disruption_active = True
            self.current_disruption = random.choice(self.disruption_types[1:])
        else:
            self.disruption_active = False
            self.current_disruption = 'none'
    
    def _update_competitor_prices(self):
        """Update competitor prices with realistic random walk"""
        for airline in self.competitor_prices:
            # Random walk with small changes
            change = np.random.normal(0, self.price_std * 0.05)
            self.competitor_prices[airline] += change
            
            # Keep prices in realistic range
            self.competitor_prices[airline] = np.clip(
                self.competitor_prices[airline],
                self.price_min,
                self.price_max
            )
    
    def step(self, action):
        """Execute one step in the environment"""
        # Map action to price adjustment
        price_changes = [-0.20, -0.10, 0.0, 0.10, 0.20]
        price_adjustment = price_changes[action]
        
        # Update price
        old_price = self.current_price
        self.current_price *= (1 + price_adjustment)
        self.current_price = np.clip(self.current_price, self.price_min, self.price_max)
        
        # Update competitors
        self._update_competitor_prices()
        
        # Trigger random disruptions
        self._trigger_disruption()
        
        # Calculate demand and bookings
        demand = self._calculate_demand()
        bookings = self._simulate_bookings(demand)
        
        # Update state
        self.seats_sold += bookings
        revenue_this_step = bookings * self.current_price
        self.total_revenue += revenue_this_step
        
        # Calculate reward
        reward = self._calculate_reward(bookings, revenue_this_step)
        
        # Update time
        self.days_to_departure -= 1
        self.current_step += 1
        
        # Check if done
        done = (self.days_to_departure <= 0) or (self.seats_sold >= self.total_seats)
        
        # Info
        info = {
            'seats_sold': self.seats_sold,
            'total_revenue': self.total_revenue,
            'current_price': self.current_price,
            'bookings': bookings,
            'disruption': self.current_disruption,
            'load_factor': self.seats_sold / self.total_seats,
            'competitor_avg': np.mean(list(self.competitor_prices.values()))
        }
        
        return self._get_state(), reward, done, info
    
    def _calculate_reward(self, bookings, revenue):
        """
        Calculate reward with realistic business objectives:
        - Maximize revenue
        - Maintain good load factor (80-85% target)
        - Penalize empty seats near departure
        """
        reward = revenue / 1000  # Scale down
        
        # Load factor bonus (target 80-85%)
        load_factor = self.seats_sold / self.total_seats
        if load_factor >= 0.80:
            reward += 5
        
        # Penalty for empty seats near departure
        if self.days_to_departure < 7:
            empty_seats = self.total_seats - self.seats_sold
            if load_factor < 0.6:
                reward -= empty_seats * 0.5
        
        # Disruption handling penalty
        if self.disruption_active:
            if self.current_disruption == 'pilot_strike':
                reward -= 10
            elif self.current_disruption == 'weather':
                reward -= 5
        
        return reward
    
    def render(self, mode='human'):
        """Render environment state"""
        print(f"\n=== Day {self.max_days - self.days_to_departure} ===")
        print(f"Route: {self.route}")
        print(f"Days to Departure: {self.days_to_departure}")
        print(f"Current Price: ₹{self.current_price:.0f}")
        print(f"Competitor Avg: ₹{np.mean(list(self.competitor_prices.values())):.0f}")
        print(f"Seats Sold: {self.seats_sold}/{self.total_seats}")
        print(f"Load Factor: {(self.seats_sold/self.total_seats)*100:.1f}%")
        print(f"Total Revenue: ₹{self.total_revenue:.0f}")
        print(f"Disruption: {self.current_disruption}")
    
    @classmethod
    def from_data(cls, filepath, route):
        """
        Create environment from flight data file
        Automatically calibrates using real data
        """
        print(f"\n🔧 Creating calibrated environment for {route}...")
        df = pd.read_csv(filepath)
        return cls(route=route, flight_data=df)


# Example usage
if __name__ == "__main__":
    print("="*70)
    print("  TESTING CALIBRATED ENVIRONMENT")
    print("="*70)
    
    # Test with data file if it exists
    if os.path.exists('data/flight_data.csv'):
        print("\n✓ Found real flight data!")
        env = AirlineRevenueEnv.from_data('data/flight_data.csv', 'Delhi-Mumbai')
    else:
        print("\n⚠️  No flight data found, using defaults")
        env = AirlineRevenueEnv(route='Delhi-Mumbai')
    
    # Test environment
    state = env.reset()
    print(f"\nInitial State: {state}")
    
    print("\n🎮 Running 5 test steps...")
    for i in range(5):
        action = env.action_space.sample()
        state, reward, done, info = env.step(action)
        env.render()
        print(f"Reward: {reward:.2f}")
        
        if done:
            print("\n✓ Episode finished!")
            break
    
    print("\n" + "="*70)
    print("  ✓ ENVIRONMENT TEST COMPLETE")
    print("="*70)