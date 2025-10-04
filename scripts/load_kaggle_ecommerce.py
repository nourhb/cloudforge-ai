# scripts/load_kaggle_ecommerce.py - Kaggle E-commerce Dataset Loader
import pandas as pd
import boto3
import json
import requests
from datetime import datetime, timedelta
import numpy as np

def load_ecommerce_data():
    """Load and process Kaggle e-commerce dataset for MinIO testing"""
    
    data_file = 'data/kaggle-ecommerce/2019-Oct.csv'
    
    try:
        # Load e-commerce data
        df = pd.read_csv(data_file)
        print(f"Loaded {len(df)} e-commerce transactions")
        
        # Basic data analysis
        print(f"Date range: {df['event_time'].min()} to {df['event_time'].max()}")
        print(f"Unique products: {df['product_id'].nunique()}")
        print(f"Unique users: {df['user_id'].nunique()}")
        print(f"Event types: {df['event_type'].value_counts()}")
        
        # Test MinIO integration
        test_minio_integration(df)
        
        # Test marketplace API
        test_marketplace_api(df)
        
        return df
        
    except FileNotFoundError:
        print(f"Error: {data_file} not found. Please download from Kaggle first.")
        return None
    except Exception as e:
        print(f"Error loading e-commerce data: {e}")
        return None

def test_minio_integration(df):
    """Test MinIO object storage with e-commerce data"""
    
    try:
        # Configure MinIO client
        from minio import Minio
        
        client = Minio(
            'localhost:9000',
            access_key='minioadmin',
            secret_key='minioadmin',
            secure=False
        )
        
        # Create bucket if not exists
        bucket_name = 'ecommerce-data'
        if not client.bucket_exists(bucket_name):
            client.make_bucket(bucket_name)
            print(f"Created bucket: {bucket_name}")
        
        # Upload sample data files
        sample_data = df.head(10000)
        
        # Save as JSON for object storage testing
        json_file = 'temp_ecommerce_sample.json'
        sample_data.to_json(json_file, orient='records', lines=True)
        
        # Upload to MinIO
        client.fput_object(
            bucket_name,
            'samples/ecommerce_oct_sample.json',
            json_file
        )
        
        print("Successfully uploaded sample data to MinIO")
        
        # Clean up temp file
        import os
        os.remove(json_file)
        
        # List objects to verify
        objects = client.list_objects(bucket_name, prefix='samples/')
        for obj in objects:
            print(f"Object in MinIO: {obj.object_name} ({obj.size} bytes)")
            
    except Exception as e:
        print(f"MinIO integration test failed: {e}")
        print("Note: Ensure MinIO is running on localhost:9000")

def test_marketplace_api(df):
    """Test marketplace API with e-commerce product data"""
    
    try:
        # Prepare product data from e-commerce dataset
        products = df.groupby('product_id').agg({
            'price': 'mean',
            'category_code': 'first',
            'brand': 'first',
            'event_type': 'count'
        }).reset_index()
        
        products = products.head(50)  # Test with 50 products
        
        # Test creating marketplace entries
        api_url = 'http://localhost:3001/api/marketplace/templates'
        
        for _, product in products.iterrows():
            template_data = {
                'name': f"Template for {product['brand']} - {product['product_id']}",
                'description': f"Infrastructure template based on {product['category_code']} usage patterns",
                'category': 'ecommerce',
                'price': float(product['price']) if pd.notna(product['price']) else 29.99,
                'rating': min(5.0, max(1.0, product['event_type'] / 100)),
                'downloads': int(product['event_type']),
                'tags': ['ecommerce', 'analytics', product['category_code']] if pd.notna(product['category_code']) else ['ecommerce'],
                'author': 'CloudForge AI',
                'compatibility': ['AWS', 'Azure', 'GCP'],
                'infrastructure_type': 'kubernetes'
            }
            
            response = requests.post(api_url, json=template_data)
            
            if response.status_code in [200, 201]:
                print(f"Created marketplace template: {template_data['name']}")
            else:
                print(f"Failed to create template: {response.status_code}")
                break  # Stop on first error
            
        print(f"Marketplace API test completed with {len(products)} products")
        
    except requests.exceptions.ConnectionError:
        print("Warning: Backend API not running. Skipping marketplace test.")
    except Exception as e:
        print(f"Marketplace API test failed: {e}")

def generate_performance_metrics(df):
    """Generate performance metrics for testing"""
    
    # Simulate system metrics based on e-commerce load
    metrics = []
    
    # Group by hour for realistic load patterns
    df['hour'] = pd.to_datetime(df['event_time']).dt.hour
    hourly_load = df.groupby('hour').size()
    
    for hour, load in hourly_load.items():
        # Simulate realistic system metrics
        cpu_usage = min(90, 20 + (load / 1000))
        memory_usage = min(85, 30 + (load / 1500))
        disk_io = min(1000, load / 10)
        network_io = min(2000, load / 5)
        
        metrics.append({
            'timestamp': datetime.now().replace(hour=hour).isoformat(),
            'cpu_usage': cpu_usage,
            'memory_usage': memory_usage,
            'disk_io': disk_io,
            'network_io': network_io,
            'response_time': max(50, 200 - (90 - cpu_usage) * 2),
            'error_rate': max(0, (cpu_usage - 70) / 20) if cpu_usage > 70 else 0
        })
    
    return metrics

if __name__ == "__main__":
    df = load_ecommerce_data()
    if df is not None:
        metrics = generate_performance_metrics(df)
        print(f"Generated {len(metrics)} performance metrics for testing")