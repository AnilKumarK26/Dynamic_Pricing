"""
Real Flight Data Preprocessing - IMPROVED VERSION
File: utils/preprocessing.py
Uses ACTUAL flight data with MULTI-FACTOR analysis to calibrate the environment
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
import pickle
import os


class FlightDataProcessor:
    def __init__(self):
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.route_stats = {}
        
    def load_data(self, filepath='data/flight_data.csv'):
        """Load REAL flight dataset"""
        if not os.path.exists(filepath):
            raise FileNotFoundError(
                f"\n❌ Flight data not found at: {filepath}\n"
                f"Please add your flight_data.csv to the data/ folder!\n"
                f"Required columns: airline, from, to, route, price, duration_in_min, stops, class_category"
            )
        
        df = pd.read_csv(filepath)
        print(f"✓ Loaded {len(df)} flight records from real data")
        
        # Validate required columns
        required_cols = ['airline', 'price']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Create route column if missing
        if 'route' not in df.columns:
            if 'from' in df.columns and 'to' in df.columns:
                df['route'] = df['from'] + '-' + df['to']
                print("✓ Created 'route' column from 'from' and 'to'")
            else:
                raise ValueError("Cannot determine routes - need 'route' or 'from'+'to' columns")
        
        return df
    
    def get_available_routes(self, df):
        """Get all unique routes in the dataset"""
        routes = df['route'].value_counts()
        print("\n📍 Available Routes:")
        for route, count in routes.items():
            print(f"   {route}: {count} flights")
        
        return routes.index.tolist()
    
    def analyze_route(self, df, route):
        """
        Analyze a specific route with MULTI-FACTOR segmentation
        This filters to get COMPARABLE base prices (apples-to-apples)
        """
        route_df = df[df['route'] == route].copy()
        
        if len(route_df) == 0:
            raise ValueError(f"No data found for route: {route}")
        
        print(f"\n{'='*80}")
        print(f"🔍 MULTI-FACTOR ANALYSIS: {route}")
        print(f"{'='*80}")
        print(f"   Total flights in dataset: {len(route_df)}")
        
        # ===== STEP 1: Filter to BASE SEGMENT (comparable flights) =====
        base_segment = route_df.copy()
        filters_applied = []
        
        # Filter 1: Non-stop flights only (most comparable)
        if 'stops' in route_df.columns:
            non_stop = base_segment[base_segment['stops'] == 0]
            if len(non_stop) > 0:
                base_segment = non_stop
                filters_applied.append(f"Non-stop flights ({len(non_stop)})")
            else:
                filters_applied.append("All stops (no non-stop data)")
        
        # Filter 2: Economy class only
        if 'class_category' in route_df.columns:
            economy = base_segment[base_segment['class_category'] == 'Economy']
            if len(economy) > 0:
                base_segment = economy
                filters_applied.append(f"Economy class ({len(economy)})")
            else:
                filters_applied.append("All classes (no economy data)")
        
        # Filter 3: Daytime departures (most common business scenario)
        if 'dep_period' in route_df.columns:
            daytime_keywords = ['Morning', 'Afternoon', 'morning', 'afternoon']
            daytime = base_segment[base_segment['dep_period'].isin(daytime_keywords)]
            if len(daytime) > 10:  # Need enough data
                base_segment = daytime
                filters_applied.append(f"Daytime departures ({len(daytime)})")
            else:
                filters_applied.append("All times (limited daytime data)")
        elif 'dep_daytime_category' in route_df.columns:
            daytime = base_segment[base_segment['dep_daytime_category'] == 'Daytime Departure']
            if len(daytime) > 10:
                base_segment = daytime
                filters_applied.append(f"Daytime departures ({len(daytime)})")
        
        print(f"\n📊 Base Segment Filters Applied:")
        for i, f in enumerate(filters_applied, 1):
            print(f"   {i}. {f}")
        print(f"\n✓ Final base segment: {len(base_segment)} comparable flights")
        
        if len(base_segment) < 5:
            print(f"\n⚠️  WARNING: Only {len(base_segment)} flights in base segment!")
            print(f"   Falling back to all flights for this route")
            base_segment = route_df
        
        # ===== STEP 2: Calculate REALISTIC Price Statistics =====
        price_stats = {
            'mean': float(base_segment['price'].mean()),
            'median': float(base_segment['price'].median()),
            'std': float(base_segment['price'].std()),
            'min': float(base_segment['price'].min()),
            'max': float(base_segment['price'].max()),
            'q10': float(base_segment['price'].quantile(0.10)),
            'q25': float(base_segment['price'].quantile(0.25)),
            'q75': float(base_segment['price'].quantile(0.75)),
            'q90': float(base_segment['price'].quantile(0.90)),
            'sample_size': len(base_segment)
        }
        
        print(f"\n💰 Price Statistics (Base Segment - Comparable Flights):")
        print(f"   Mean:       ₹{price_stats['mean']:.0f}")
        print(f"   Median:     ₹{price_stats['median']:.0f}")
        print(f"   Std Dev:    ₹{price_stats['std']:.0f}")
        print(f"   Range:      ₹{price_stats['min']:.0f} - ₹{price_stats['max']:.0f}")
        print(f"   IQR (Q25-Q75): ₹{price_stats['q25']:.0f} - ₹{price_stats['q75']:.0f}")
        print(f"   Sample:     {price_stats['sample_size']} flights")
        
        # ===== STEP 3: Get COMPETITOR PRICES (from base segment) =====
        print(f"\n🏢 Competitor Analysis (Base Segment):")
        
        competitor_prices = {}
        competitor_details = {}
        
        for airline in base_segment['airline'].unique():
            airline_data = base_segment[base_segment['airline'] == airline]
            
            if len(airline_data) == 0:
                continue
            
            # Use MEDIAN (more robust to outliers than mean)
            median_price = float(airline_data['price'].median())
            mean_price = float(airline_data['price'].mean())
            
            competitor_prices[airline] = median_price
            
            competitor_details[airline] = {
                'median': median_price,
                'mean': mean_price,
                'count': len(airline_data),
                'std': float(airline_data['price'].std()) if len(airline_data) > 1 else 0,
                'min': float(airline_data['price'].min()),
                'max': float(airline_data['price'].max())
            }
            
            print(f"   {airline:15s} → Median: ₹{median_price:6.0f}, "
                  f"Mean: ₹{mean_price:6.0f}, "
                  f"Count: {len(airline_data):3d}, "
                  f"Std: ₹{competitor_details[airline]['std']:.0f}")
        
        # ===== STEP 4: Calculate PRICE MODIFIERS (for simulation variations) =====
        price_modifiers = self._calculate_price_modifiers(route_df)
        
        if price_modifiers:
            print(f"\n🎛️  Price Modifiers (for simulation variations):")
            if 'stops' in price_modifiers:
                print(f"   Stops impact:")
                for stops, factor in price_modifiers['stops'].items():
                    print(f"      {stops} stops: {factor:.2f}x base price")
            
            if 'time_of_day' in price_modifiers:
                print(f"   Time of day impact:")
                for period, factor in sorted(price_modifiers['time_of_day'].items(), 
                                            key=lambda x: x[1], reverse=True)[:3]:
                    print(f"      {period}: {factor:.2f}x base price")
        
        # ===== STEP 5: Additional Statistics =====
        # Airlines on ALL flights (for reference)
        airlines_all = route_df['airline'].value_counts()
        print(f"\n✈️  All Airlines on Route (all flight types):")
        for airline, count in airlines_all.items():
            avg_price = route_df[route_df['airline'] == airline]['price'].mean()
            print(f"   {airline}: {count} flights (overall avg: ₹{avg_price:.0f})")
        
        # Duration statistics (if available)
        duration_stats = None
        if 'duration_in_min' in route_df.columns:
            duration_stats = {
                'mean': float(route_df['duration_in_min'].mean()),
                'median': float(route_df['duration_in_min'].median()),
                'std': float(route_df['duration_in_min'].std()),
                'min': float(route_df['duration_in_min'].min()),
                'max': float(route_df['duration_in_min'].max())
            }
            print(f"\n⏱️  Duration Statistics:")
            print(f"   Mean: {duration_stats['mean']:.0f} min")
            print(f"   Range: {duration_stats['min']:.0f} - {duration_stats['max']:.0f} min")
        
        # Stops analysis (if available)
        if 'stops' in route_df.columns:
            stops_dist = route_df['stops'].value_counts(normalize=True).sort_index()
            print(f"\n🛬 Stops Distribution:")
            for stops, pct in stops_dist.items():
                count = len(route_df[route_df['stops'] == stops])
                avg_price = route_df[route_df['stops'] == stops]['price'].mean()
                print(f"   {stops} stops: {pct*100:5.1f}% ({count} flights, avg: ₹{avg_price:.0f})")
        
        # Class distribution (if available)
        if 'class_category' in route_df.columns:
            class_dist = route_df['class_category'].value_counts(normalize=True)
            print(f"\n💺 Class Distribution:")
            for cls, pct in class_dist.items():
                count = len(route_df[route_df['class_category'] == cls])
                avg_price = route_df[route_df['class_category'] == cls]['price'].mean()
                print(f"   {cls}: {pct*100:5.1f}% ({count} flights, avg: ₹{avg_price:.0f})")
        
        # ===== STEP 6: Store comprehensive statistics =====
        route_stats = {
            'route': route,
            'n_flights': len(route_df),
            'n_base_flights': len(base_segment),
            'filters_applied': filters_applied,
            'price_stats': price_stats,
            'competitor_prices': competitor_prices,
            'competitor_details': competitor_details,
            'price_modifiers': price_modifiers,
            'airlines': airlines_all.to_dict(),
            'duration_stats': duration_stats,
        }
        
        self.route_stats[route] = route_stats
        
        print(f"\n{'='*80}")
        print(f"✓ Analysis complete for {route}")
        print(f"{'='*80}")
        
        return route_stats
    
    def _calculate_price_modifiers(self, route_df):
        """
        Calculate price adjustment factors for different conditions
        These help simulate realistic price variations in the RL environment
        """
        modifiers = {}
        
        # Stops modifier - how much more expensive are connecting flights?
        if 'stops' in route_df.columns and 0 in route_df['stops'].values:
            base_price = route_df[route_df['stops'] == 0]['price'].median()
            if not pd.isna(base_price) and base_price > 0:
                modifiers['stops'] = {}
                for stops in route_df['stops'].unique():
                    stop_data = route_df[route_df['stops'] == stops]
                    if len(stop_data) > 0:
                        stop_price = stop_data['price'].median()
                        modifiers['stops'][int(stops)] = float(stop_price / base_price)
        
        # Time of day modifier - are evening/night flights cheaper?
        if 'dep_period' in route_df.columns:
            overall_median = route_df['price'].median()
            if overall_median > 0:
                modifiers['time_of_day'] = {}
                for period in route_df['dep_period'].unique():
                    period_data = route_df[route_df['dep_period'] == period]
                    if len(period_data) > 0:
                        period_price = period_data['price'].median()
                        modifiers['time_of_day'][period] = float(period_price / overall_median)
        
        # Class modifier - how much more expensive is Business/First?
        if 'class_category' in route_df.columns:
            economy_data = route_df[route_df['class_category'] == 'Economy']
            if len(economy_data) > 0:
                economy_price = economy_data['price'].median()
                if not pd.isna(economy_price) and economy_price > 0:
                    modifiers['class'] = {}
                    for class_cat in route_df['class_category'].unique():
                        class_data = route_df[route_df['class_category'] == class_cat]
                        if len(class_data) > 0:
                            class_price = class_data['price'].median()
                            modifiers['class'][class_cat] = float(class_price / economy_price)
        
        return modifiers
    
    def get_calibrated_env_params(self, route_stats):
        """
        Convert route statistics to environment parameters
        This ensures the RL environment uses REAL data patterns!
        """
        price_stats = route_stats['price_stats']
        
        params = {
            # Base pricing from REAL BASE SEGMENT data (comparable flights)
            'base_price': price_stats['median'],  # Use median (more robust)
            'price_mean': price_stats['mean'],
            'price_std': price_stats['std'],
            
            # Realistic price bounds
            'price_min': price_stats['q25'],  # 25th percentile (competitive low)
            'price_max': price_stats['q75'] * 1.3,  # 30% above 75th percentile (surge)
            
            # Competitor prices from REAL BASE SEGMENT data
            'competitor_prices': route_stats['competitor_prices'],
            'competitor_details': route_stats['competitor_details'],
            
            # Price modifiers for simulation variations
            'price_modifiers': route_stats.get('price_modifiers', {}),
            
            # Realistic aircraft & booking parameters
            'total_seats': 180,  # Standard single-aisle aircraft (A320/B737)
            'max_days_before_departure': 90,
            
            # Demand parameters (calibrated to realistic booking patterns)
            'base_demand_rate': 0.15,  # 15% of capacity per day on average
            'price_elasticity': 2.0,  # How sensitive demand is to price changes
            
            # Time-based booking curve (more bookings closer to departure)
            'early_booking_factor': 0.3,  # Lower demand 90 days out
            'late_booking_factor': 1.5,   # Higher demand close to departure
            
            # Route metadata
            'route': route_stats['route'],
            'n_airlines': len(route_stats['competitor_prices']),
            'sample_size': price_stats['sample_size']
        }
        
        print(f"\n⚙️  Calibrated Environment Parameters:")
        print(f"   Route: {params['route']}")
        print(f"   Base Price (median): ₹{params['base_price']:.0f}")
        print(f"   Price Mean: ₹{params['price_mean']:.0f}")
        print(f"   Price Range: ₹{params['price_min']:.0f} - ₹{params['price_max']:.0f}")
        print(f"   Competitors: {params['n_airlines']}")
        print(f"   Sample Size: {params['sample_size']} comparable flights")
        print(f"   Total Seats: {params['total_seats']}")
        
        return params
    
    def preprocess(self, df):
        """Preprocess flight data for RL"""
        df_processed = df.copy()
        
        # Encode categorical variables
        categorical_cols = ['airline', 'from', 'to', 'route']
        
        for col in categorical_cols:
            if col in df_processed.columns:
                le = LabelEncoder()
                df_processed[col + '_encoded'] = le.fit_transform(df_processed[col].astype(str))
                self.label_encoders[col] = le
        
        # Feature engineering
        if 'duration_in_min' in df_processed.columns:
            df_processed['price_per_minute'] = df_processed['price'] / df_processed['duration_in_min']
        
        if 'dep_hour' in df_processed.columns:
            df_processed['is_morning_flight'] = (df_processed['dep_hour'] >= 6) & (df_processed['dep_hour'] < 12)
            df_processed['is_evening_flight'] = (df_processed['dep_hour'] >= 18) & (df_processed['dep_hour'] < 24)
        
        return df_processed
    
    def save_route_stats(self, filepath='data/route_stats.pkl'):
        """Save analyzed route statistics"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(self.route_stats, f)
        print(f"\n✓ Route statistics saved to {filepath}")
    
    def load_route_stats(self, filepath='data/route_stats.pkl'):
        """Load analyzed route statistics"""
        if os.path.exists(filepath):
            with open(filepath, 'rb') as f:
                self.route_stats = pickle.load(f)
            print(f"✓ Route statistics loaded from {filepath}")
            return True
        else:
            print(f"⚠️  No saved route statistics found at {filepath}")
            return False


# Main execution for data analysis
if __name__ == "__main__":
    print("="*80)
    print("  REAL FLIGHT DATA ANALYSIS - MULTI-FACTOR VERSION")
    print("="*80)
    
    try:
        processor = FlightDataProcessor()
        
        # Load REAL data
        df = processor.load_data('data/flight_data.csv')
        
        # Get available routes
        routes = processor.get_available_routes(df)
        
        # Analyze first route (or you can specify)
        if routes:
            selected_route = routes[0]  # Change this to analyze different routes
            print(f"\n🎯 Selected Route: {selected_route}")
            
            # Analyze route with MULTI-FACTOR analysis
            route_stats = processor.analyze_route(df, selected_route)
            
            # Get calibrated parameters
            env_params = processor.get_calibrated_env_params(route_stats)
            
            # Save statistics
            processor.save_route_stats()
            
            print("\n" + "="*80)
            print("  ✓ MULTI-FACTOR ANALYSIS COMPLETE")
            print("="*80)
            print("\n📊 Key Improvements:")
            print("   1. ✓ Filtered to COMPARABLE flights (non-stop, economy, daytime)")
            print("   2. ✓ Used MEDIAN prices (robust to outliers)")
            print("   3. ✓ Calculated price modifiers for variations")
            print("   4. ✓ Realistic base prices for RL environment")
            print("\nUse these parameters in your environment for realistic simulation!")
            
        else:
            print("\n❌ No routes found in data!")
            
    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        print("\n💡 Solution:")
        print("   1. Add your flight_data.csv to the data/ folder")
        print("   2. Ensure it has columns: airline, route, price")
        print("   3. Run this script again")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()