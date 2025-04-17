import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
import argparse

def generate_synthetic_price_data(crop_name, start_date, days, base_price, volatility, trend=0, seasonality=False):
    """
    Generate synthetic price data for a crop.
    
    Parameters:
    crop_name (str): Name of the crop.
    start_date (datetime): Starting date for the data.
    days (int): Number of days to generate data for.
    base_price (float): Base price for the crop.
    volatility (float): Volatility factor for the price (0-1).
    trend (float): Price trend factor (e.g., 0.01 for 1% daily increase).
    seasonality (bool): Whether to add seasonal patterns.
    
    Returns:
    DataFrame: Generated price data.
    """
    dates = [start_date + timedelta(days=i) for i in range(days)]
    
    # Generate base prices with random fluctuations
    prices = []
    current_price = base_price
    
    for i in range(days):
        # Add trend
        current_price *= (1 + trend)
        
        # Add seasonality (if enabled)
        seasonal_factor = 0
        if seasonality:
            # Simple sine wave seasonality with 365-day period
            seasonal_factor = np.sin(2 * np.pi * i / 365) * 0.15
        
        # Add random fluctuation
        random_factor = np.random.normal(0, volatility)
        
        # Calculate new price
        new_price = current_price * (1 + random_factor + seasonal_factor)
        
        # Ensure price doesn't go negative
        new_price = max(0.1, new_price)
        
        prices.append(new_price)
        current_price = new_price
    
    # Create DataFrame
    df = pd.DataFrame({
        'date': dates,
        'crop_name': crop_name,
        'price': prices
    })
    
    return df

def generate_market_data(crop_name, markets, base_price, market_factors):
    """
    Generate price data for multiple markets.
    
    Parameters:
    crop_name (str): Name of the crop.
    markets (list): List of market locations.
    base_price (float): Base price for the crop.
    market_factors (dict): Price factors for each market.
    
    Returns:
    DataFrame: Generated price data for all markets.
    """
    all_data = []
    
    for market in markets:
        # Calculate market-specific base price
        market_base_price = base_price * market_factors.get(market, 1.0)
        
        # Generate data for this market
        market_data = generate_synthetic_price_data(
            crop_name=crop_name,
            start_date=datetime(2023, 1, 1),
            days=365,  # Generate one year of data
            base_price=market_base_price,
            volatility=0.02,  # 2% daily volatility
            trend=0.0002,  # Very slight upward trend (0.02% daily)
            seasonality=True
        )
        
        # Add market location
        market_data['market_location'] = market
        all_data.append(market_data)
    
    # Combine all market data
    combined_data = pd.concat(all_data, ignore_index=True)
    return combined_data

def main():
    parser = argparse.ArgumentParser(description='Generate synthetic crop price data.')
    parser.add_argument('-o', '--output', type=str, default='historical_prices.csv',
                        help='Output CSV filename')
    args = parser.parse_args()
    
    print("Generating synthetic crop price data...")
    
    # Define crop types
    crops = {
        'rice': {'base_price': 25.0},
        'wheat': {'base_price': 22.0},
        'maize': {'base_price': 18.0},
        'cotton': {'base_price': 65.0},
        'sugarcane': {'base_price': 3.5}
    }
    
    # Define market locations and their price factors
    markets = ['Mumbai', 'Delhi', 'Kolkata', 'Chennai', 'Bangalore']
    market_factors = {
        'Mumbai': 1.05,
        'Delhi': 1.00,
        'Kolkata': 0.95,
        'Chennai': 1.02,
        'Bangalore': 1.03
    }
    
    all_crop_data = []
    
    # Generate data for each crop
    for crop_name, crop_info in crops.items():
        print(f"Generating data for {crop_name}...")
        crop_data = generate_market_data(
            crop_name=crop_name,
            markets=markets,
            base_price=crop_info['base_price'],
            market_factors=market_factors
        )
        all_crop_data.append(crop_data)
    
    # Combine all crop data
    combined_data = pd.concat(all_crop_data, ignore_index=True)
    
    # Format date
    combined_data['date'] = combined_data['date'].dt.strftime('%Y-%m-%d')
    
    # Round prices to 2 decimal places
    combined_data['price'] = combined_data['price'].round(2)
    
    # Save to CSV
    combined_data.to_csv(args.output, index=False)
    print(f"Data saved to {args.output}")
    print(f"Generated {len(combined_data)} price records for {len(crops)} crops in {len(markets)} markets")

if __name__ == "__main__":
    main()