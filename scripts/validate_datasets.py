# scripts/validate_datasets.py - Dataset Validation and Testing
import os
import json
import pandas as pd
import sqlite3
import psycopg2
import requests
from datetime import datetime

def validate_chinook_db():
    """Validate Chinook database integration"""
    print("=== Validating Chinook Database ===")
    
    try:
        # Test SQLite connection
        conn = sqlite3.connect('data/chinook/chinook.db')
        cursor = conn.cursor()
        
        # Validate tables and record counts
        tables = ['albums', 'artists', 'customers', 'employees', 'genres', 
                 'invoice_items', 'invoices', 'media_types', 'playlist_track', 
                 'playlists', 'tracks']
        
        total_records = 0
        for table in tables:
            cursor.execute(f"SELECT COUNT(*) FROM {table}")
            count = cursor.fetchone()[0]
            print(f"  {table}: {count} records")
            total_records += count
        
        print(f"Total Chinook records: {total_records}")
        
        # Test complex query
        cursor.execute("""
            SELECT g.Name as Genre, COUNT(t.TrackId) as Tracks, 
                   SUM(t.Milliseconds)/1000/60 as TotalMinutes
            FROM genres g
            JOIN tracks t ON g.GenreId = t.GenreId
            GROUP BY g.GenreId
            ORDER BY Tracks DESC
            LIMIT 5
        """)
        
        results = cursor.fetchall()
        print("Top 5 Genres by Track Count:")
        for genre, tracks, minutes in results:
            print(f"  {genre}: {tracks} tracks, {minutes:.1f} minutes")
        
        conn.close()
        return True
        
    except Exception as e:
        print(f"Chinook validation failed: {e}")
        return False

def validate_uci_dataset():
    """Validate UCI network dataset"""
    print("\n=== Validating UCI Network Dataset ===")
    
    try:
        data_file = 'data/uci-network/kddcup.data_10_percent'
        if not os.path.exists(data_file):
            print(f"UCI dataset not found: {data_file}")
            return False
            
        # Count lines without loading full dataset
        with open(data_file, 'r') as f:
            line_count = sum(1 for line in f)
        
        print(f"UCI Network records: {line_count}")
        
        # Load sample for validation
        df = pd.read_csv(data_file, nrows=1000, header=None)
        print(f"Sample loaded: {len(df)} records, {len(df.columns)} features")
        
        # Check for attack types in sample
        attack_labels = df.iloc[:, -1].unique()
        print(f"Attack types in sample: {len(attack_labels)}")
        print(f"Sample labels: {list(attack_labels)[:5]}")
        
        return True
        
    except Exception as e:
        print(f"UCI dataset validation failed: {e}")
        return False

def validate_kaggle_ecommerce():
    """Validate Kaggle e-commerce dataset"""
    print("\n=== Validating Kaggle E-commerce Dataset ===")
    
    try:
        data_file = 'data/kaggle-ecommerce/2019-Oct.csv'
        if not os.path.exists(data_file):
            print(f"E-commerce dataset not found: {data_file}")
            return False
        
        # Load sample for validation
        df = pd.read_csv(data_file, nrows=10000)
        print(f"E-commerce sample: {len(df)} records")
        print(f"Columns: {list(df.columns)}")
        
        # Validate required columns
        required_cols = ['event_time', 'event_type', 'product_id', 'price', 'user_id']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            print(f"Missing required columns: {missing_cols}")
            return False
        
        # Basic statistics
        print(f"Date range: {df['event_time'].min()} to {df['event_time'].max()}")
        print(f"Event types: {df['event_type'].value_counts().to_dict()}")
        print(f"Price range: ${df['price'].min():.2f} - ${df['price'].max():.2f}")
        
        return True
        
    except Exception as e:
        print(f"E-commerce validation failed: {e}")
        return False

def test_ai_services():
    """Test AI services with real data"""
    print("\n=== Testing AI Services ===")
    
    test_results = {}
    
    # Test anomaly detection
    try:
        # Generate realistic metrics for testing
        test_data = {
            'metrics_data': [
                {
                    'timestamp': datetime.now().isoformat(),
                    'cpu_usage': 85.5,
                    'memory_usage': 72.3,
                    'disk_io': 450,
                    'network_io': 890
                },
                {
                    'timestamp': datetime.now().isoformat(),
                    'cpu_usage': 15.2,
                    'memory_usage': 45.1,
                    'disk_io': 120,
                    'network_io': 230
                }
            ]
        }
        
        response = requests.post('http://localhost:5001/ai/detect/anomalies', 
                               json=test_data, timeout=10)
        
        if response.status_code == 200:
            result = response.json()
            test_results['anomaly_detection'] = 'PASS'
            print(f"✓ Anomaly Detection: {result.get('total_anomalies', 0)} anomalies detected")
        else:
            test_results['anomaly_detection'] = f'FAIL: {response.status_code}'
            print(f"✗ Anomaly Detection failed: {response.status_code}")
            
    except requests.exceptions.ConnectionError:
        test_results['anomaly_detection'] = 'FAIL: Service not running'
        print("✗ Anomaly Detection: AI services not running")
    except Exception as e:
        test_results['anomaly_detection'] = f'FAIL: {str(e)}'
        print(f"✗ Anomaly Detection error: {e}")
    
    # Test forecasting
    try:
        forecast_data = {
            'metric_name': 'cpu_usage',
            'historical_data': [45.2, 52.1, 48.9, 55.3, 49.7, 53.2, 47.8],
            'periods': 3
        }
        
        response = requests.post('http://localhost:5001/ai/forecast', 
                               json=forecast_data, timeout=10)
        
        if response.status_code == 200:
            result = response.json()
            test_results['forecasting'] = 'PASS'
            print(f"✓ Forecasting: Generated {len(result.get('forecast', []))} predictions")
        else:
            test_results['forecasting'] = f'FAIL: {response.status_code}'
            print(f"✗ Forecasting failed: {response.status_code}")
            
    except requests.exceptions.ConnectionError:
        test_results['forecasting'] = 'FAIL: Service not running'
        print("✗ Forecasting: AI services not running")
    except Exception as e:
        test_results['forecasting'] = f'FAIL: {str(e)}'
        print(f"✗ Forecasting error: {e}")
    
    return test_results

def test_backend_api():
    """Test backend API endpoints"""
    print("\n=== Testing Backend API ===")
    
    test_results = {}
    
    # Test health endpoint
    try:
        response = requests.get('http://localhost:3001/health', timeout=5)
        if response.status_code == 200:
            test_results['health'] = 'PASS'
            print("✓ Health endpoint: OK")
        else:
            test_results['health'] = f'FAIL: {response.status_code}'
            print(f"✗ Health endpoint failed: {response.status_code}")
    except Exception as e:
        test_results['health'] = f'FAIL: {str(e)}'
        print(f"✗ Health endpoint error: {e}")
    
    # Test marketplace endpoint
    try:
        response = requests.get('http://localhost:3001/api/marketplace/templates', timeout=5)
        if response.status_code == 200:
            templates = response.json()
            test_results['marketplace'] = 'PASS'
            print(f"✓ Marketplace API: {len(templates)} templates available")
        else:
            test_results['marketplace'] = f'FAIL: {response.status_code}'
            print(f"✗ Marketplace API failed: {response.status_code}")
    except Exception as e:
        test_results['marketplace'] = f'FAIL: {str(e)}'
        print(f"✗ Marketplace API error: {e}")
    
    return test_results

def generate_validation_report():
    """Generate comprehensive validation report"""
    print("\n" + "="*60)
    print("CLOUDFORGE AI DATASET VALIDATION REPORT")
    print("="*60)
    
    report = {
        'validation_time': datetime.now().isoformat(),
        'datasets': {},
        'services': {}
    }
    
    # Validate datasets
    report['datasets']['chinook'] = validate_chinook_db()
    report['datasets']['uci_network'] = validate_uci_dataset()
    report['datasets']['kaggle_ecommerce'] = validate_kaggle_ecommerce()
    
    # Test services
    report['services']['ai_services'] = test_ai_services()
    report['services']['backend_api'] = test_backend_api()
    
    # Summary
    print(f"\n=== VALIDATION SUMMARY ===")
    dataset_pass = sum(1 for v in report['datasets'].values() if v)
    dataset_total = len(report['datasets'])
    print(f"Datasets: {dataset_pass}/{dataset_total} passed")
    
    # Save report
    with open('validation_report.json', 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"Detailed report saved to: validation_report.json")
    
    return report

if __name__ == "__main__":
    generate_validation_report()