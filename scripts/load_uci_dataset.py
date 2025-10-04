# scripts/load_uci_dataset.py - UCI Dataset Loader
import pandas as pd
import requests
import numpy as np
from datetime import datetime
import os

def load_uci_network_data():
    """Load and process UCI network intrusion dataset"""
    
    # Define column names (41 features)
    columns = [
        'duration', 'protocol_type', 'service', 'flag', 'src_bytes',
        'dst_bytes', 'land', 'wrong_fragment', 'urgent', 'hot',
        'num_failed_logins', 'logged_in', 'num_compromised', 'root_shell',
        'su_attempted', 'num_root', 'num_file_creations', 'num_shells',
        'num_access_files', 'num_outbound_cmds', 'is_host_login',
        'is_guest_login', 'count', 'srv_count', 'serror_rate',
        'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate',
        'same_srv_rate', 'diff_srv_rate', 'srv_diff_host_rate',
        'dst_host_count', 'dst_host_srv_count', 'dst_host_same_srv_rate',
        'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
        'dst_host_srv_diff_host_rate', 'dst_host_serror_rate',
        'dst_host_srv_serror_rate', 'dst_host_rerror_rate',
        'dst_host_srv_rerror_rate', 'label'
    ]
    
    # Load data
    data_file = 'data/uci-network/kddcup.data_10_percent'
    if not os.path.exists(data_file):
        print(f"Error: {data_file} not found. Please download first.")
        return None
        
    df = pd.read_csv(data_file, names=columns, header=None)
    
    print(f"Loaded {len(df)} network connection records")
    print(f"Attack types distribution:")
    print(df['label'].value_counts().head(10))
    
    # Prepare data for anomaly detection API
    normal_data = df[df['label'] == 'normal.'].head(1000)
    attack_data = df[df['label'] != 'normal.'].head(200)
    
    # Convert to format expected by CloudForge AI
    test_data = []
    for _, row in pd.concat([normal_data, attack_data]).iterrows():
        test_data.append({
            'timestamp': datetime.now().isoformat(),
            'cpu_usage': min(100, row['src_bytes'] / 1000),
            'memory_usage': min(100, row['dst_bytes'] / 1000),
            'disk_io': min(1000, row['count'] * 10),
            'network_io': min(1000, row['srv_count'] * 20),
            'is_anomaly': row['label'] != 'normal.'
        })
    
    # Test anomaly detection
    try:
        response = requests.post('http://localhost:5001/ai/detect/anomalies',
                               json={'metrics_data': test_data[:100]})
        
        if response.status_code == 200:
            results = response.json()
            print(f"Anomaly Detection Results:")
            print(f"Total anomalies detected: {results['total_anomalies']}")
            print(f"Accuracy: {results['anomaly_percentage']:.2f}%")
            return results
        else:
            print(f"Error testing anomaly detection: {response.status_code}")
            return None
    except requests.exceptions.ConnectionError:
        print("Warning: AI services not running. Skipping API test.")
        return None

if __name__ == "__main__":
    load_uci_network_data()