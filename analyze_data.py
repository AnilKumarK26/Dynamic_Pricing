"""
Analyze Real Flight Data and Calibrate Environment
Run this FIRST to understand your data and set up the environment
UPDATED: Now uses multi-factor price analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from utils.preprocessing import FlightDataProcessor
import os

def main():
    print("="*80)
    print("  REAL FLIGHT DATA ANALYSIS & CALIBRATION")
    print("  Multi-Factor Price Segmentation")
    print("="*80)
    
    # Check if data exists
    data_path = 'data/flight_data.csv'
    if not os.path.exists(data_path):
        print(f"\n❌ ERROR: No flight data found at {data_path}")
        print("\n📋 Required Steps:")
        print("   1. Add your CSV file to: data/flight_data.csv")
        print("   2. Ensure it has these columns:")
        print("      - airline (e.g., 'SpiceJet', 'IndiGo')")
        print("      - from (e.g., 'Delhi')")
        print("      - to (e.g., 'Mumbai')")
        print("      - route (e.g., 'Delhi-Mumbai') OR we'll create it")
        print("      - price (e.g., 5953)")
        print("   3. Optional columns: duration_in_min, stops, class_category, dep_period")
        print("\n💡 Your data format example:")
        print("   airline,from,to,price,duration_in_min,stops,class_category")
        print("   SpiceJet,Delhi,Mumbai,5953,130,0,Economy")
        print("   AirAsia,Delhi,Mumbai,5956,130,0,Economy")
        return
    
    # Load processor
    processor = FlightDataProcessor()
    
    try:
        # Load data
        print("\n📂 Loading flight data...")
        df = processor.load_data(data_path)
        
        # Show data info
        print(f"\n📊 Dataset Overview:")
        print(f"   Total Records: {len(df):,}")
        print(f"   Columns: {', '.join(df.columns)}")
        
        # Check for important columns
        important_cols = ['stops', 'class_category', 'dep_period', 'duration_in_min']
        available_cols = [col for col in important_cols if col in df.columns]
        missing_cols = [col for col in important_cols if col not in df.columns]
        
        if available_cols:
            print(f"   ✓ Available for segmentation: {', '.join(available_cols)}")
        if missing_cols:
            print(f"   ⚠️ Missing (optional): {', '.join(missing_cols)}")
        
        # Get available routes
        print("\n" + "-"*80)
        routes = processor.get_available_routes(df)
        
        if not routes:
            print("\n❌ No routes found in data!")
            return
        
        # Let user select route or use first one
        print(f"\n🎯 Found {len(routes)} routes")
        print("\nSelect a route to analyze:")
        for i, route in enumerate(routes[:10], 1):  # Show first 10
            count = len(df[df['route'] == route])
            avg_price = df[df['route'] == route]['price'].mean()
            print(f"   {i}. {route:25s} ({count:4d} flights, avg: ₹{avg_price:.0f})")
        
        if len(routes) > 10:
            print(f"   ... and {len(routes)-10} more routes")
        
        # For now, analyze the route with most data
        selected_route = df['route'].value_counts().index[0]
        print(f"\n✨ Auto-selecting route with most data: {selected_route}")
        
        print("\n" + "="*80)
        
        # MULTI-FACTOR ANALYSIS of selected route
        route_stats = processor.analyze_route(df, selected_route)
        
        # Get calibrated parameters
        print("\n" + "-"*80)
        env_params = processor.get_calibrated_env_params(route_stats)
        
        # Save statistics
        processor.save_route_stats()
        
        # Create visualizations
        print("\n📊 Creating visualizations...")
        create_route_visualizations(df, selected_route, route_stats)
        
        # Summary and recommendations
        print("\n" + "="*80)
        print("  ✅ ANALYSIS COMPLETE!")
        print("="*80)
        
        print("\n📈 Key Insights:")
        print(f"   Route: {selected_route}")
        print(f"   Base Price (Median): ₹{route_stats['price_stats']['median']:.0f}")
        print(f"   Base Price (Mean): ₹{route_stats['price_stats']['mean']:.0f}")
        print(f"   Price Std Dev: ₹{route_stats['price_stats']['std']:.0f}")
        print(f"   Price Range: ₹{route_stats['price_stats']['min']:.0f} - ₹{route_stats['price_stats']['max']:.0f}")
        print(f"   Competitors: {len(route_stats['competitor_prices'])}")
        print(f"   Total Flights in Data: {route_stats['n_flights']}")
        print(f"   Base Segment Flights: {route_stats['n_base_flights']}")
        
        print("\n🏢 Competitor Base Prices (from comparable flights):")
        for airline, price in sorted(route_stats['competitor_prices'].items(), 
                                     key=lambda x: x[1]):
            details = route_stats['competitor_details'][airline]
            print(f"   {airline:15s} → ₹{price:6.0f} (n={details['count']})")
        
        print("\n🎯 Next Steps:")
        print("   1. ✓ Route statistics saved to data/route_stats.pkl")
        print("   2. ✓ Visualizations saved to results/")
        print("   3. Run: python app.py (to start web dashboard)")
        print("   4. Run: python main.py --route '{}'".format(selected_route))
        print("      (to train RL agent on this route)")
        
        print("\n💡 The environment is now calibrated with YOUR real data!")
        print("   All price ranges, competitor behavior, and demand patterns")
        print("   are based on COMPARABLE flights (non-stop, economy, daytime)")
        print("   from your dataset.")
        
    except Exception as e:
        print(f"\n❌ Error during analysis: {e}")
        import traceback
        traceback.print_exc()
        
        print("\n💡 Troubleshooting:")
        print("   - Check your CSV format")
        print("   - Ensure price column has numeric values")
        print("   - Verify airline and route columns exist")


def create_route_visualizations(df, route, route_stats):
    """Create comprehensive visualizations for the route"""
    route_df = df[df['route'] == route]
    
    # Create results directory
    os.makedirs('results', exist_ok=True)
    
    # Set style
    sns.set_style('darkgrid')
    
    # Determine number of subplots based on available data
    n_plots = 6
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    fig.suptitle(f'Multi-Factor Flight Analysis: {route}', 
                 fontsize=18, fontweight='bold')
    
    # 1. Overall Price Distribution
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.hist(route_df['price'], bins=30, edgecolor='black', alpha=0.7, color='skyblue')
    ax1.axvline(route_stats['price_stats']['mean'], color='red', 
                linestyle='--', linewidth=2, 
                label=f"Mean: ₹{route_stats['price_stats']['mean']:.0f}")
    ax1.axvline(route_stats['price_stats']['median'], color='green', 
                linestyle='--', linewidth=2, 
                label=f"Median: ₹{route_stats['price_stats']['median']:.0f}")
    ax1.set_xlabel('Price (₹)', fontsize=11)
    ax1.set_ylabel('Frequency', fontsize=11)
    ax1.set_title('Price Distribution (All Flights)', fontsize=13, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Price by Airline (Base Segment)
    ax2 = fig.add_subplot(gs[0, 1])
    comp_prices = route_stats['competitor_prices']
    comp_details = route_stats['competitor_details']
    
    airlines = list(comp_prices.keys())
    medians = [comp_prices[a] for a in airlines]
    counts = [comp_details[a]['count'] for a in airlines]
    
    # Sort by price
    sorted_data = sorted(zip(airlines, medians, counts), key=lambda x: x[1])
    airlines_sorted = [x[0] for x in sorted_data]
    medians_sorted = [x[1] for x in sorted_data]
    counts_sorted = [x[2] for x in sorted_data]
    
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(airlines_sorted)))
    bars = ax2.barh(airlines_sorted, medians_sorted, color=colors, alpha=0.8)
    
    ax2.set_xlabel('Median Price (₹)', fontsize=11)
    ax2.set_title('Competitor Prices (Base Segment)', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='x')
    
    # Add count labels
    for i, (bar, count) in enumerate(zip(bars, counts_sorted)):
        ax2.text(bar.get_width(), bar.get_y() + bar.get_height()/2, 
                f' ₹{medians_sorted[i]:.0f} (n={count})',
                va='center', fontsize=9)
    
    # 3. Price by Number of Stops
    ax3 = fig.add_subplot(gs[0, 2])
    if 'stops' in route_df.columns:
        stops_data = route_df.groupby('stops')['price'].agg(['mean', 'median', 'count'])
        colors_stops = ['#2ecc71', '#f39c12', '#e74c3c'][:len(stops_data)]
        
        x = np.arange(len(stops_data))
        width = 0.35
        
        ax3.bar(x - width/2, stops_data['mean'], width, label='Mean', 
                alpha=0.8, color='steelblue')
        ax3.bar(x + width/2, stops_data['median'], width, label='Median', 
                alpha=0.8, color='coral')
        
        ax3.set_xlabel('Number of Stops', fontsize=11)
        ax3.set_ylabel('Price (₹)', fontsize=11)
        ax3.set_title('Price by Number of Stops', fontsize=13, fontweight='bold')
        ax3.set_xticks(x)
        ax3.set_xticklabels(stops_data.index)
        ax3.legend()
        ax3.grid(True, alpha=0.3, axis='y')
        
        # Add count labels
        for i, (idx, row) in enumerate(stops_data.iterrows()):
            ax3.text(i, row['mean'], f"n={row['count']}", 
                    ha='center', va='bottom', fontsize=8)
    else:
        ax3.text(0.5, 0.5, 'Stops data not available', 
                ha='center', va='center', fontsize=12, transform=ax3.transAxes)
        ax3.set_title('Price by Number of Stops', fontsize=13, fontweight='bold')
    
    # 4. Price by Time of Day
    ax4 = fig.add_subplot(gs[1, 0])
    if 'dep_period' in route_df.columns:
        time_data = route_df.groupby('dep_period')['price'].agg(['mean', 'median', 'count']).sort_values('median')
        
        ax4.barh(time_data.index, time_data['median'], color='coral', alpha=0.7, label='Median')
        ax4.set_xlabel('Median Price (₹)', fontsize=11)
        ax4.set_title('Price by Departure Time', fontsize=13, fontweight='bold')
        ax4.grid(True, alpha=0.3, axis='x')
        
        for i, (idx, row) in enumerate(time_data.iterrows()):
            ax4.text(row['median'], i, f" ₹{row['median']:.0f} (n={row['count']})", 
                    va='center', fontsize=9)
    else:
        ax4.text(0.5, 0.5, 'Time data not available', 
                ha='center', va='center', fontsize=12, transform=ax4.transAxes)
        ax4.set_title('Price by Departure Time', fontsize=13, fontweight='bold')
    
    # 5. Price by Class
    ax5 = fig.add_subplot(gs[1, 1])
    if 'class_category' in route_df.columns:
        class_data = route_df.groupby('class_category')['price'].agg(['mean', 'median', 'count']).sort_values('median')
        
        colors_class = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(class_data)))
        bars = ax5.bar(class_data.index, class_data['median'], color=colors_class, alpha=0.8)
        
        ax5.set_xlabel('Class', fontsize=11)
        ax5.set_ylabel('Median Price (₹)', fontsize=11)
        ax5.set_title('Price by Class Category', fontsize=13, fontweight='bold')
        ax5.grid(True, alpha=0.3, axis='y')
        
        for i, (bar, (idx, row)) in enumerate(zip(bars, class_data.iterrows())):
            ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                    f"₹{row['median']:.0f}\n(n={row['count']})",
                    ha='center', va='bottom', fontsize=9)
    else:
        ax5.text(0.5, 0.5, 'Class data not available', 
                ha='center', va='center', fontsize=12, transform=ax5.transAxes)
        ax5.set_title('Price by Class Category', fontsize=13, fontweight='bold')
    
    # 6. Price vs Duration
    ax6 = fig.add_subplot(gs[1, 2])
    if 'duration_in_min' in route_df.columns:
        scatter = ax6.scatter(route_df['duration_in_min'], route_df['price'], 
                             alpha=0.4, c=route_df['price'], cmap='viridis', s=30)
        ax6.set_xlabel('Duration (minutes)', fontsize=11)
        ax6.set_ylabel('Price (₹)', fontsize=11)
        ax6.set_title('Price vs Duration', fontsize=13, fontweight='bold')
        plt.colorbar(scatter, ax=ax6, label='Price (₹)')
        ax6.grid(True, alpha=0.3)
        
        # Add trend line
        valid_mask = route_df['duration_in_min'].notna() & route_df['price'].notna()
        if valid_mask.sum() > 2:
            z = np.polyfit(route_df.loc[valid_mask, 'duration_in_min'], 
                          route_df.loc[valid_mask, 'price'], 1)
            p = np.poly1d(z)
            x_line = np.linspace(route_df['duration_in_min'].min(), 
                                route_df['duration_in_min'].max(), 100)
            ax6.plot(x_line, p(x_line), "r--", alpha=0.8, linewidth=2, label='Trend')
            ax6.legend()
    else:
        ax6.text(0.5, 0.5, 'Duration data not available', 
                ha='center', va='center', fontsize=12, transform=ax6.transAxes)
        ax6.set_title('Price vs Duration', fontsize=13, fontweight='bold')
    
    # 7. Base Segment Statistics
    ax7 = fig.add_subplot(gs[2, :2])
    stats_text = f"""
ROUTE: {route}

BASE SEGMENT (Comparable Flights):
  • Sample Size: {route_stats['n_base_flights']} flights
  • Filters: {', '.join(route_stats['filters_applied'])}

PRICE STATISTICS (Base Segment):
  • Median:  ₹{route_stats['price_stats']['median']:.0f}
  • Mean:    ₹{route_stats['price_stats']['mean']:.0f}
  • Std Dev: ₹{route_stats['price_stats']['std']:.0f}
  • Range:   ₹{route_stats['price_stats']['min']:.0f} - ₹{route_stats['price_stats']['max']:.0f}
  • IQR:     ₹{route_stats['price_stats']['q25']:.0f} - ₹{route_stats['price_stats']['q75']:.0f}

CALIBRATED ENVIRONMENT:
  • Base Price:   ₹{route_stats['price_stats']['median']:.0f}
  • Price Floor:  ₹{route_stats['price_stats']['q25']:.0f}
  • Price Ceiling: ₹{route_stats['price_stats']['q75'] * 1.3:.0f}
  • Competitors:  {len(route_stats['competitor_prices'])}

ALL FLIGHTS:
  • Total Flights: {route_stats['n_flights']}
  • Airlines: {len(route_stats['airlines'])}
    """
    
    ax7.text(0.05, 0.5, stats_text, fontsize=11, verticalalignment='center',
            family='monospace', 
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    ax7.axis('off')
    ax7.set_title('Calibration Summary', fontsize=13, fontweight='bold', pad=10)
    
    # 8. Competitor Details Table
    ax8 = fig.add_subplot(gs[2, 2])
    ax8.axis('tight')
    ax8.axis('off')
    
    table_data = []
    table_data.append(['Airline', 'Median', 'Count', 'Std'])
    for airline in sorted(route_stats['competitor_prices'].keys(), 
                         key=lambda x: route_stats['competitor_prices'][x]):
        details = route_stats['competitor_details'][airline]
        table_data.append([
            airline[:12],  # Truncate long names
            f"₹{details['median']:.0f}",
            f"{details['count']}",
            f"₹{details['std']:.0f}"
        ])
    
    table = ax8.table(cellText=table_data, cellLoc='left', loc='center',
                     colWidths=[0.4, 0.2, 0.2, 0.2])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)
    
    # Style header row
    for i in range(4):
        table[(0, i)].set_facecolor('#40466e')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    ax8.set_title('Competitor Details (Base Segment)', 
                 fontsize=13, fontweight='bold', pad=20)
    
    plt.tight_layout()
    
    # Save
    safe_route = route.replace("/", "_").replace("\\", "_")
    save_path = f'results/route_analysis_{safe_route}.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"   ✓ Saved visualization: {save_path}")
    
    plt.close()


if __name__ == "__main__":
    main()