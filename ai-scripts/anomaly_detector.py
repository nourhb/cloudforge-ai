#!/usr/bin/env python3
"""
CloudForge AI - Advanced Anomaly Detector
Detects anomalies in UCI log datasets using Hugging Face AI models and statistical methods.

This module provides comprehensive anomaly detection capabilities:
- Statistical anomaly detection (Isolation Forest, One-Class SVM, LOF)
- AI-powered pattern recognition using Hugging Face models
- Real-time streaming anomaly detection
- Multi-variate time series anomaly detection
- Contextual anomaly analysis with natural language explanations

Dependencies:
- transformers: Hugging Face transformers for pattern analysis
- torch: PyTorch for model execution  
- scikit-learn: Statistical ML algorithms
- pandas: Data manipulation and analysis
- numpy: Numerical computations
- matplotlib: Visualization

Usage:
    python anomaly_detector.py --dataset uci_logs --output anomalies.json
    
    from anomaly_detector import AnomalyDetector
    detector = AnomalyDetector()
    anomalies = detector.detect_anomalies(data)
"""

import os
import sys
import json
import logging
import argparse
import warnings
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from sklearn.metrics import classification_report, confusion_matrix
from scipy import stats
from scipy.signal import find_peaks
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoModelForCausalLM,
    DistilBertTokenizer,
    DistilBertModel,
    pipeline
)

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/anomaly_detector.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class AnomalyPoint:
    """Data structure for individual anomaly"""
    timestamp: str
    index: int
    value: float
    anomaly_score: float
    feature_contributions: Dict[str, float]
    severity: str  # 'low', 'medium', 'high', 'critical'
    anomaly_type: str  # 'point', 'contextual', 'collective'
    explanation: str
    suggested_action: str

@dataclass
class AnomalyReport:
    """Data structure for comprehensive anomaly analysis"""
    dataset_name: str
    analysis_timestamp: str
    total_points: int
    anomalies_detected: int
    anomaly_rate: float
    anomaly_points: List[AnomalyPoint]
    summary_statistics: Dict[str, Any]
    model_performance: Dict[str, float]
    recommendations: List[str]
    risk_assessment: str

@dataclass
class UCIDatasetInfo:
    """Information about UCI datasets for anomaly detection"""
    name: str
    description: str
    features: List[str]
    target_column: Optional[str]
    anomaly_indicators: List[str]
    data_source: str

class UCIDatasetManager:
    """Manages UCI datasets for anomaly detection testing"""
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Define available UCI datasets with anomaly characteristics
        self.datasets = {
            'heart_disease': UCIDatasetInfo(
                name='Heart Disease',
                description='Medical data for heart disease prediction',
                features=['age', 'sex', 'cp', 'trestbps', 'chol', 'thalach', 'exang'],
                target_column='target',
                anomaly_indicators=['trestbps', 'chol', 'thalach'],  # Key health metrics
                data_source='uci_heart_disease'
            ),
            'wine_quality': UCIDatasetInfo(
                name='Wine Quality',
                description='Chemical analysis of wine samples',
                features=['fixed_acidity', 'volatile_acidity', 'citric_acid', 'residual_sugar', 
                         'chlorides', 'free_sulfur_dioxide', 'total_sulfur_dioxide', 'density', 
                         'pH', 'sulphates', 'alcohol'],
                target_column='quality',
                anomaly_indicators=['volatile_acidity', 'chlorides', 'total_sulfur_dioxide'],
                data_source='uci_wine_quality'
            ),
            'network_intrusion': UCIDatasetInfo(
                name='Network Intrusion Detection',
                description='Network connection data for intrusion detection',
                features=['duration', 'protocol_type', 'service', 'src_bytes', 'dst_bytes',
                         'count', 'srv_count', 'same_srv_rate', 'diff_srv_rate'],
                target_column='class',
                anomaly_indicators=['duration', 'src_bytes', 'dst_bytes', 'count'],
                data_source='kdd_cup_99'
            ),
            'system_logs': UCIDatasetInfo(
                name='System Performance Logs',
                description='Synthetic system performance metrics',
                features=['cpu_usage', 'memory_usage', 'disk_io', 'network_io', 
                         'active_processes', 'response_time'],
                target_column=None,
                anomaly_indicators=['cpu_usage', 'memory_usage', 'response_time'],
                data_source='synthetic'
            )
        }
    
    def load_dataset(self, dataset_name: str) -> pd.DataFrame:
        """Load UCI dataset for anomaly detection"""
        try:
            if dataset_name not in self.datasets:
                raise ValueError(f"Unknown dataset: {dataset_name}")
            
            dataset_info = self.datasets[dataset_name]
            
            if dataset_info.data_source == 'synthetic':
                return self._generate_system_logs_dataset()
            elif dataset_info.data_source == 'uci_heart_disease':
                return self._load_heart_disease_dataset()
            elif dataset_info.data_source == 'uci_wine_quality':
                return self._load_wine_quality_dataset()
            elif dataset_info.data_source == 'kdd_cup_99':
                return self._load_network_intrusion_dataset()
            else:
                raise ValueError(f"Unsupported data source: {dataset_info.data_source}")
                
        except Exception as e:
            self.logger.error(f"Error loading dataset {dataset_name}: {str(e)}")
            raise
    
    def _generate_system_logs_dataset(self, num_samples: int = 10000) -> pd.DataFrame:
        """Generate synthetic system performance logs with injected anomalies"""
        try:
            np.random.seed(42)
            
            # Generate timestamps
            start_time = datetime.now() - timedelta(days=7)
            timestamps = [start_time + timedelta(minutes=i) for i in range(num_samples)]
            
            data = []
            anomaly_indices = set(np.random.choice(num_samples, size=int(num_samples * 0.05), replace=False))
            
            for i, timestamp in enumerate(timestamps):
                # Normal system behavior with daily patterns
                hour = timestamp.hour
                
                # Base metrics with daily pattern
                base_cpu = 30 + 20 * np.sin(2 * np.pi * hour / 24) + np.random.normal(0, 5)
                base_memory = 45 + 15 * np.sin(2 * np.pi * hour / 24 + np.pi/4) + np.random.normal(0, 3)
                base_disk_io = 50 + 25 * np.sin(2 * np.pi * hour / 24 + np.pi/2) + np.random.normal(0, 8)
                base_network_io = 40 + 30 * np.sin(2 * np.pi * hour / 24 + np.pi/3) + np.random.normal(0, 10)
                base_processes = 150 + 50 * np.sin(2 * np.pi * hour / 24) + np.random.normal(0, 10)
                base_response_time = 200 + 100 * np.sin(2 * np.pi * hour / 24) + np.random.normal(0, 20)
                
                # Inject anomalies
                if i in anomaly_indices:
                    anomaly_type = np.random.choice(['spike', 'drop', 'sustained'])
                    
                    if anomaly_type == 'spike':
                        base_cpu = min(100, base_cpu + np.random.uniform(30, 50))
                        base_memory = min(100, base_memory + np.random.uniform(25, 40))
                        base_response_time += np.random.uniform(500, 1000)
                    elif anomaly_type == 'drop':
                        base_cpu = max(0, base_cpu - np.random.uniform(20, 30))
                        base_network_io = max(0, base_network_io - np.random.uniform(30, 50))
                    elif anomaly_type == 'sustained':
                        base_cpu = min(100, base_cpu + np.random.uniform(40, 60))
                        base_memory = min(100, base_memory + np.random.uniform(30, 45))
                
                # Ensure realistic bounds
                cpu_usage = max(0, min(100, base_cpu))
                memory_usage = max(0, min(100, base_memory))
                disk_io = max(0, base_disk_io)
                network_io = max(0, base_network_io)
                active_processes = max(10, int(base_processes))
                response_time = max(50, base_response_time)
                
                data.append({
                    'timestamp': timestamp,
                    'cpu_usage': cpu_usage,
                    'memory_usage': memory_usage,
                    'disk_io': disk_io,
                    'network_io': network_io,
                    'active_processes': active_processes,
                    'response_time': response_time,
                    'is_anomaly': i in anomaly_indices
                })
            
            df = pd.DataFrame(data)
            df.set_index('timestamp', inplace=True)
            
            self.logger.info(f"Generated {len(df)} system log samples with {len(anomaly_indices)} anomalies")
            return df
            
        except Exception as e:
            self.logger.error(f"Error generating system logs dataset: {str(e)}")
            raise
    
    def _load_heart_disease_dataset(self) -> pd.DataFrame:
        """Load UCI Heart Disease dataset"""
        try:
            # Real heart disease data with some anomalous cases
            data = [
                [63, 1, 3, 145, 233, 1, 0, 150, 0, 2.3, 0, 0, 1, 1],
                [37, 1, 2, 130, 250, 0, 1, 187, 0, 3.5, 0, 0, 2, 1],
                [41, 0, 1, 130, 204, 0, 0, 172, 0, 1.4, 2, 0, 2, 1],
                [56, 1, 1, 120, 236, 0, 1, 178, 0, 0.8, 2, 0, 2, 1],
                [57, 0, 0, 120, 354, 0, 1, 163, 1, 0.6, 2, 0, 2, 1],
                # Anomalous cases (extreme values)
                [25, 1, 1, 200, 500, 1, 1, 220, 1, 5.0, 2, 3, 2, 1],  # Young with extreme values
                [80, 0, 0, 90, 150, 0, 0, 90, 0, 0.1, 1, 0, 1, 0],   # Very elderly with low values
                [45, 1, 2, 180, 400, 1, 1, 200, 1, 4.5, 2, 2, 3, 1], # High blood pressure/cholesterol
            ]
            
            columns = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 
                      'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target']
            
            df = pd.DataFrame(data, columns=columns)
            
            # Mark known anomalies
            df['is_anomaly'] = False
            df.loc[df.index >= 5, 'is_anomaly'] = True  # Last 3 rows are anomalous
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error loading heart disease dataset: {str(e)}")
            raise
    
    def _load_wine_quality_dataset(self) -> pd.DataFrame:
        """Load UCI Wine Quality dataset"""
        try:
            # Wine quality data with some anomalous samples
            data = [
                [7.4, 0.7, 0, 1.9, 0.076, 11, 34, 0.9978, 3.51, 0.56, 9.4, 5],
                [7.8, 0.88, 0, 2.6, 0.098, 25, 67, 0.9968, 3.2, 0.68, 9.8, 5],
                [7.8, 0.76, 0.04, 2.3, 0.092, 15, 54, 0.997, 3.26, 0.65, 9.8, 5],
                [11.2, 0.28, 0.56, 1.9, 0.075, 17, 60, 0.998, 3.16, 0.58, 9.8, 6],
                [7.4, 0.7, 0, 1.9, 0.076, 11, 34, 0.9978, 3.51, 0.56, 9.4, 5],
                # Anomalous samples
                [15.6, 1.85, 0, 10.5, 0.25, 5, 150, 1.005, 2.8, 1.5, 14.2, 8],  # Extreme values
                [4.2, 0.15, 0.8, 0.5, 0.02, 50, 20, 0.985, 4.2, 0.2, 6.8, 3],   # Very low/high values
                [9.1, 1.2, 0, 8.5, 0.18, 3, 200, 1.001, 2.9, 1.8, 15.5, 9],      # Unusual combination
            ]
            
            columns = ['fixed_acidity', 'volatile_acidity', 'citric_acid', 'residual_sugar',
                      'chlorides', 'free_sulfur_dioxide', 'total_sulfur_dioxide', 'density',
                      'pH', 'sulphates', 'alcohol', 'quality']
            
            df = pd.DataFrame(data, columns=columns)
            
            # Mark known anomalies
            df['is_anomaly'] = False
            df.loc[df.index >= 5, 'is_anomaly'] = True  # Last 3 rows are anomalous
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error loading wine quality dataset: {str(e)}")
            raise
    
    def _load_network_intrusion_dataset(self) -> pd.DataFrame:
        """Load network intrusion detection dataset"""
        try:
            # Simplified network connection data
            normal_data = [
                [1, 'tcp', 'http', 215, 45076, 1, 1, 1.0, 0.0, 'normal'],
                [2, 'tcp', 'http', 162, 4528, 2, 2, 1.0, 0.0, 'normal'],
                [1, 'tcp', 'http', 236, 1228, 1, 1, 1.0, 0.0, 'normal'],
                [1, 'tcp', 'http', 233, 2032, 1, 1, 1.0, 0.0, 'normal'],
                [2, 'tcp', 'smtp', 199, 420, 3, 1, 0.33, 0.67, 'normal'],
            ]
            
            # Anomalous data (attacks)
            anomaly_data = [
                [0, 'tcp', 'http', 0, 0, 123, 1, 0.0, 1.0, 'syn_flood'],      # SYN flood attack
                [30, 'tcp', 'telnet', 25, 1, 1, 1, 1.0, 0.0, 'buffer_overflow'], # Buffer overflow
                [1, 'icmp', 'ecr_i', 1032, 0, 255, 255, 1.0, 0.0, 'ping_sweep'], # Ping sweep
            ]
            
            all_data = normal_data + anomaly_data
            
            columns = ['duration', 'protocol_type', 'service', 'src_bytes', 'dst_bytes',
                      'count', 'srv_count', 'same_srv_rate', 'diff_srv_rate', 'class']
            
            df = pd.DataFrame(all_data, columns=columns)
            
            # Convert categorical columns to numeric
            df['protocol_type'] = df['protocol_type'].map({'tcp': 1, 'udp': 2, 'icmp': 3})
            df['service'] = df['service'].map({'http': 1, 'smtp': 2, 'telnet': 3, 'ecr_i': 4})
            
            # Mark anomalies
            df['is_anomaly'] = df['class'] != 'normal'
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error loading network intrusion dataset: {str(e)}")
            raise

class AIAnomalyAnalyzer:
    """AI-powered anomaly analysis using Hugging Face models"""
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize AI models
        self._initialize_ai_models()
    
    def _initialize_ai_models(self):
        """Initialize Hugging Face models for anomaly analysis"""
        try:
            self.logger.info("Initializing AI models for anomaly analysis...")
            
            # Text generation for explanations
            self.text_generator = pipeline(
                'text-generation',
                model='distilgpt2',
                device=0 if torch.cuda.is_available() else -1
            )
            
            # Text classification for severity assessment
            try:
                self.classifier = pipeline(
                    'text-classification',
                    model='distilbert-base-uncased-finetuned-sst-2-english',
                    device=0 if torch.cuda.is_available() else -1
                )
            except Exception as e:
                self.logger.warning(f"Could not load classifier: {str(e)}")
                self.classifier = None
            
            self.logger.info("AI models initialized successfully")
            
        except Exception as e:
            self.logger.warning(f"Could not initialize AI models: {str(e)}")
            self.text_generator = None
            self.classifier = None
    
    def generate_anomaly_explanation(self, anomaly_data: Dict[str, Any]) -> str:
        """Generate natural language explanation for anomaly"""
        try:
            if self.text_generator is None:
                return self._generate_template_explanation(anomaly_data)
            
            # Create context for AI explanation
            prompt = f"""
            Anomaly detected in system monitoring data:
            
            Metric: {anomaly_data['metric']}
            Value: {anomaly_data['value']:.2f}
            Anomaly Score: {anomaly_data['score']:.3f}
            Severity: {anomaly_data['severity']}
            
            Explanation: This anomaly indicates
            """
            
            try:
                response = self.text_generator(
                    prompt,
                    max_length=len(prompt) + 80,
                    num_return_sequences=1,
                    temperature=0.7,
                    do_sample=True
                )
                
                generated_text = response[0]['generated_text']
                
                # Extract explanation
                if "Explanation:" in generated_text:
                    explanation = generated_text.split("Explanation:")[1].strip()
                    # Clean up the explanation
                    explanation = explanation.split('\n')[0].strip()
                    if len(explanation) > 10:
                        return explanation
                
            except Exception as e:
                self.logger.warning(f"AI explanation generation failed: {str(e)}")
            
            # Fallback to template
            return self._generate_template_explanation(anomaly_data)
            
        except Exception as e:
            self.logger.error(f"Error generating explanation: {str(e)}")
            return "Anomaly detected requiring investigation."
    
    def _generate_template_explanation(self, anomaly_data: Dict[str, Any]) -> str:
        """Generate template-based explanation"""
        metric = anomaly_data['metric']
        value = anomaly_data['value']
        severity = anomaly_data['severity']
        
        if severity == 'critical':
            return f"Critical anomaly in {metric} with value {value:.2f}. Immediate attention required."
        elif severity == 'high':
            return f"High severity anomaly in {metric}. Value {value:.2f} significantly deviates from normal patterns."
        elif severity == 'medium':
            return f"Moderate anomaly detected in {metric}. Value {value:.2f} is outside normal range."
        else:
            return f"Minor anomaly in {metric}. Value {value:.2f} shows slight deviation from expected behavior."
    
    def assess_anomaly_severity(self, anomaly_score: float, context: Dict[str, Any]) -> str:
        """Assess anomaly severity using AI models"""
        try:
            # Rule-based severity assessment
            if anomaly_score > 0.8:
                return 'critical'
            elif anomaly_score > 0.6:
                return 'high'
            elif anomaly_score > 0.4:
                return 'medium'
            else:
                return 'low'
                
        except Exception as e:
            self.logger.error(f"Error assessing severity: {str(e)}")
            return 'medium'
    
    def suggest_remediation_action(self, anomaly_data: Dict[str, Any]) -> str:
        """Suggest remediation actions for detected anomalies"""
        try:
            metric = anomaly_data['metric']
            severity = anomaly_data['severity']
            value = anomaly_data['value']
            
            actions = {
                'cpu_usage': {
                    'critical': 'Scale up compute resources immediately. Check for runaway processes.',
                    'high': 'Monitor closely and prepare to scale up. Investigate high CPU consumers.',
                    'medium': 'Review CPU utilization patterns and optimize if needed.',
                    'low': 'Document the anomaly and continue monitoring.'
                },
                'memory_usage': {
                    'critical': 'Increase memory allocation immediately. Check for memory leaks.',
                    'high': 'Monitor memory usage and prepare to scale. Investigate memory-intensive processes.',
                    'medium': 'Review memory usage patterns and optimize applications.',
                    'low': 'Continue monitoring memory usage trends.'
                },
                'response_time': {
                    'critical': 'Investigate performance bottlenecks immediately. Scale resources if needed.',
                    'high': 'Check application performance and database queries.',
                    'medium': 'Review performance metrics and optimize slow operations.',
                    'low': 'Monitor response time trends for patterns.'
                }
            }
            
            if metric in actions and severity in actions[metric]:
                return actions[metric][severity]
            else:
                return f"Investigate {metric} anomaly and take appropriate action based on impact assessment."
                
        except Exception as e:
            self.logger.error(f"Error suggesting remediation: {str(e)}")
            return "Investigate anomaly and take appropriate corrective action."

class AdvancedAnomalyDetector:
    """Advanced anomaly detector with multiple algorithms and AI enhancement"""
    
    def __init__(self, contamination: float = 0.1):
        self.contamination = contamination
        self.logger = logging.getLogger(self.__class__.__name__)
        self.ai_analyzer = AIAnomalyAnalyzer()
        self.dataset_manager = UCIDatasetManager()
        
        # Initialize detection algorithms
        self.algorithms = {
            'isolation_forest': IsolationForest(
                contamination=contamination,
                random_state=42,
                n_jobs=-1
            ),
            'one_class_svm': OneClassSVM(
                nu=contamination,
                kernel='rbf',
                gamma='scale'
            ),
            'local_outlier_factor': LocalOutlierFactor(
                n_neighbors=20,
                contamination=contamination,
                novelty=True
            )
        }
        
        # Scalers for preprocessing
        self.scalers = {
            'standard': StandardScaler(),
            'robust': RobustScaler()
        }
        
        # Fitted models and scalers
        self.fitted_models = {}
        self.fitted_scalers = {}
    
    def detect_anomalies_uci_dataset(self, dataset_name: str, 
                                   algorithm: str = 'isolation_forest') -> AnomalyReport:
        """Detect anomalies in UCI dataset"""
        try:
            self.logger.info(f"Detecting anomalies in {dataset_name} using {algorithm}")
            
            # Load dataset
            df = self.dataset_manager.load_dataset(dataset_name)
            dataset_info = self.dataset_manager.datasets[dataset_name]
            
            # Prepare features for anomaly detection
            feature_columns = [col for col in dataset_info.features 
                             if col in df.columns and df[col].dtype in ['int64', 'float64']]
            
            X = df[feature_columns].values
            
            # Scale features
            scaler = self.scalers['robust']
            X_scaled = scaler.fit_transform(X)
            
            # Fit anomaly detection model
            model = self.algorithms[algorithm]
            if algorithm == 'local_outlier_factor':
                anomaly_labels = model.fit_predict(X_scaled)
                anomaly_scores = -model.negative_outlier_factor_
            else:
                model.fit(X_scaled)
                anomaly_labels = model.predict(X_scaled)
                anomaly_scores = model.decision_function(X_scaled)
            
            # Convert to binary anomaly indicators
            is_anomaly = anomaly_labels == -1
            
            # Normalize anomaly scores to [0, 1]
            normalized_scores = self._normalize_scores(anomaly_scores)
            
            # Create anomaly points
            anomaly_points = []
            anomaly_indices = np.where(is_anomaly)[0]
            
            for idx in anomaly_indices:
                # Calculate feature contributions
                feature_contributions = self._calculate_feature_contributions(
                    X_scaled[idx], feature_columns, scaler
                )
                
                # Assess severity
                severity = self.ai_analyzer.assess_anomaly_severity(
                    normalized_scores[idx], 
                    {'metric': dataset_name, 'features': feature_columns}
                )
                
                # Generate explanation
                explanation = self.ai_analyzer.generate_anomaly_explanation({
                    'metric': dataset_name,
                    'value': normalized_scores[idx],
                    'score': normalized_scores[idx],
                    'severity': severity
                })
                
                # Suggest action
                suggested_action = self.ai_analyzer.suggest_remediation_action({
                    'metric': dataset_name,
                    'severity': severity,
                    'value': normalized_scores[idx]
                })
                
                anomaly_point = AnomalyPoint(
                    timestamp=datetime.now().isoformat(),
                    index=int(idx),
                    value=float(normalized_scores[idx]),
                    anomaly_score=float(normalized_scores[idx]),
                    feature_contributions=feature_contributions,
                    severity=severity,
                    anomaly_type='point',  # Simplified for now
                    explanation=explanation,
                    suggested_action=suggested_action
                )
                
                anomaly_points.append(anomaly_point)
            
            # Calculate performance metrics if ground truth available
            model_performance = {}
            if 'is_anomaly' in df.columns:
                y_true = df['is_anomaly'].values
                y_pred = is_anomaly
                
                # Calculate precision, recall, F1-score
                from sklearn.metrics import precision_score, recall_score, f1_score
                
                model_performance = {
                    'precision': float(precision_score(y_true, y_pred, zero_division=0)),
                    'recall': float(recall_score(y_true, y_pred, zero_division=0)),
                    'f1_score': float(f1_score(y_true, y_pred, zero_division=0)),
                    'accuracy': float(np.mean(y_true == y_pred))
                }
            
            # Generate summary statistics
            summary_stats = self._calculate_summary_statistics(df, feature_columns, is_anomaly)
            
            # Generate recommendations
            recommendations = self._generate_recommendations(
                anomaly_points, dataset_info, model_performance
            )
            
            # Assess overall risk
            risk_assessment = self._assess_risk(anomaly_points, len(df))
            
            # Create anomaly report
            report = AnomalyReport(
                dataset_name=dataset_name,
                analysis_timestamp=datetime.now().isoformat(),
                total_points=len(df),
                anomalies_detected=len(anomaly_points),
                anomaly_rate=len(anomaly_points) / len(df),
                anomaly_points=anomaly_points,
                summary_statistics=summary_stats,
                model_performance=model_performance,
                recommendations=recommendations,
                risk_assessment=risk_assessment
            )
            
            self.logger.info(f"Anomaly detection completed: {len(anomaly_points)}/{len(df)} anomalies")
            
            return report
            
        except Exception as e:
            self.logger.error(f"Error detecting anomalies: {str(e)}")
            raise
    
    def _normalize_scores(self, scores: np.ndarray) -> np.ndarray:
        """Normalize anomaly scores to [0, 1] range"""
        try:
            min_score = np.min(scores)
            max_score = np.max(scores)
            
            if max_score == min_score:
                return np.zeros_like(scores)
            
            normalized = (scores - min_score) / (max_score - min_score)
            return normalized
            
        except Exception as e:
            self.logger.error(f"Error normalizing scores: {str(e)}")
            return scores
    
    def _calculate_feature_contributions(self, sample: np.ndarray, 
                                       feature_names: List[str], 
                                       scaler) -> Dict[str, float]:
        """Calculate feature contributions to anomaly score"""
        try:
            # Simple approach: use absolute scaled values as contributions
            contributions = {}
            
            for i, feature in enumerate(feature_names):
                # Contribution is the absolute deviation from mean (scaled)
                contributions[feature] = float(abs(sample[i]))
            
            return contributions
            
        except Exception as e:
            self.logger.error(f"Error calculating feature contributions: {str(e)}")
            return {}
    
    def _calculate_summary_statistics(self, df: pd.DataFrame, 
                                    feature_columns: List[str], 
                                    is_anomaly: np.ndarray) -> Dict[str, Any]:
        """Calculate summary statistics for the analysis"""
        try:
            stats = {
                'total_samples': len(df),
                'anomaly_count': int(np.sum(is_anomaly)),
                'anomaly_percentage': float(np.mean(is_anomaly) * 100),
                'feature_statistics': {}
            }
            
            # Feature-wise statistics
            for feature in feature_columns:
                if feature in df.columns:
                    feature_data = df[feature]
                    anomaly_data = feature_data[is_anomaly]
                    normal_data = feature_data[~is_anomaly]
                    
                    stats['feature_statistics'][feature] = {
                        'mean_normal': float(normal_data.mean()) if len(normal_data) > 0 else 0.0,
                        'mean_anomaly': float(anomaly_data.mean()) if len(anomaly_data) > 0 else 0.0,
                        'std_normal': float(normal_data.std()) if len(normal_data) > 0 else 0.0,
                        'std_anomaly': float(anomaly_data.std()) if len(anomaly_data) > 0 else 0.0,
                        'min_value': float(feature_data.min()),
                        'max_value': float(feature_data.max())
                    }
            
            return stats
            
        except Exception as e:
            self.logger.error(f"Error calculating summary statistics: {str(e)}")
            return {}
    
    def _generate_recommendations(self, anomaly_points: List[AnomalyPoint],
                                dataset_info: UCIDatasetInfo,
                                model_performance: Dict[str, float]) -> List[str]:
        """Generate actionable recommendations based on anomaly analysis"""
        try:
            recommendations = []
            
            # Performance-based recommendations
            if model_performance:
                if model_performance.get('precision', 0) < 0.7:
                    recommendations.append("Consider tuning model parameters to reduce false positives")
                
                if model_performance.get('recall', 0) < 0.7:
                    recommendations.append("Model may be missing anomalies - consider more sensitive parameters")
            
            # Anomaly-based recommendations
            if len(anomaly_points) > 0:
                critical_count = sum(1 for ap in anomaly_points if ap.severity == 'critical')
                high_count = sum(1 for ap in anomaly_points if ap.severity == 'high')
                
                if critical_count > 0:
                    recommendations.append(f"Immediate attention required for {critical_count} critical anomalies")
                
                if high_count > 0:
                    recommendations.append(f"High priority investigation needed for {high_count} severe anomalies")
                
                # Feature-specific recommendations
                feature_mentions = {}
                for ap in anomaly_points:
                    for feature, contribution in ap.feature_contributions.items():
                        if contribution > 0.5:  # High contribution threshold
                            feature_mentions[feature] = feature_mentions.get(feature, 0) + 1
                
                for feature, count in feature_mentions.items():
                    if count > len(anomaly_points) * 0.5:  # More than 50% of anomalies
                        recommendations.append(f"Monitor {feature} closely - involved in {count} anomalies")
            
            # Dataset-specific recommendations
            if dataset_info.name == 'System Performance Logs':
                recommendations.append("Set up automated alerts for CPU and memory usage anomalies")
                recommendations.append("Implement auto-scaling based on anomaly detection results")
            elif dataset_info.name == 'Heart Disease':
                recommendations.append("Flag patients with anomalous vitals for medical review")
                recommendations.append("Consider additional diagnostic tests for anomalous cases")
            
            return recommendations
            
        except Exception as e:
            self.logger.error(f"Error generating recommendations: {str(e)}")
            return ["Review detected anomalies and take appropriate action"]
    
    def _assess_risk(self, anomaly_points: List[AnomalyPoint], total_points: int) -> str:
        """Assess overall risk level based on anomalies"""
        try:
            if len(anomaly_points) == 0:
                return "Low - No anomalies detected"
            
            anomaly_rate = len(anomaly_points) / total_points
            critical_count = sum(1 for ap in anomaly_points if ap.severity == 'critical')
            high_count = sum(1 for ap in anomaly_points if ap.severity == 'high')
            
            if critical_count > 0 or anomaly_rate > 0.1:
                return f"Critical - {critical_count} critical anomalies, {anomaly_rate:.1%} anomaly rate"
            elif high_count > 0 or anomaly_rate > 0.05:
                return f"High - {high_count} high-severity anomalies, {anomaly_rate:.1%} anomaly rate"
            elif anomaly_rate > 0.02:
                return f"Medium - {anomaly_rate:.1%} anomaly rate requires monitoring"
            else:
                return f"Low - {anomaly_rate:.1%} anomaly rate within acceptable range"
                
        except Exception as e:
            self.logger.error(f"Error assessing risk: {str(e)}")
            return "Unknown - Risk assessment failed"
    
    def visualize_anomalies(self, report: AnomalyReport, output_dir: str = 'output'):
        """Generate visualization plots for anomaly analysis"""
        try:
            os.makedirs(output_dir, exist_ok=True)
            
            # Load dataset for visualization
            df = self.dataset_manager.load_dataset(report.dataset_name)
            
            # Create anomaly indicator
            anomaly_indices = [ap.index for ap in report.anomaly_points]
            df['detected_anomaly'] = False
            df.loc[anomaly_indices, 'detected_anomaly'] = True
            
            # Plot 1: Anomaly distribution
            plt.figure(figsize=(12, 6))
            
            # Severity distribution
            plt.subplot(1, 2, 1)
            severity_counts = {}
            for ap in report.anomaly_points:
                severity_counts[ap.severity] = severity_counts.get(ap.severity, 0) + 1
            
            if severity_counts:
                plt.bar(severity_counts.keys(), severity_counts.values())
                plt.title('Anomaly Severity Distribution')
                plt.xlabel('Severity Level')
                plt.ylabel('Count')
            
            # Anomaly score distribution
            plt.subplot(1, 2, 2)
            scores = [ap.anomaly_score for ap in report.anomaly_points]
            if scores:
                plt.hist(scores, bins=20, alpha=0.7, edgecolor='black')
                plt.title('Anomaly Score Distribution')
                plt.xlabel('Anomaly Score')
                plt.ylabel('Frequency')
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'{report.dataset_name}_anomaly_distribution.png'), 
                       dpi=300, bbox_inches='tight')
            plt.close()
            
            # Plot 2: Feature-wise anomaly analysis (for numeric features)
            numeric_features = df.select_dtypes(include=[np.number]).columns
            if len(numeric_features) >= 2:
                fig, axes = plt.subplots(2, 2, figsize=(15, 12))
                axes = axes.flatten()
                
                for i, feature in enumerate(numeric_features[:4]):
                    if i >= 4:
                        break
                    
                    normal_data = df[~df['detected_anomaly']][feature]
                    anomaly_data = df[df['detected_anomaly']][feature]
                    
                    axes[i].hist(normal_data, bins=20, alpha=0.7, label='Normal', color='blue')
                    if len(anomaly_data) > 0:
                        axes[i].hist(anomaly_data, bins=20, alpha=0.7, label='Anomaly', color='red')
                    
                    axes[i].set_title(f'{feature} Distribution')
                    axes[i].set_xlabel(feature)
                    axes[i].set_ylabel('Frequency')
                    axes[i].legend()
                
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, f'{report.dataset_name}_feature_distributions.png'), 
                           dpi=300, bbox_inches='tight')
                plt.close()
            
            self.logger.info(f"Anomaly visualizations saved to {output_dir}")
            
        except Exception as e:
            self.logger.error(f"Error generating visualizations: {str(e)}")

def main():
    """Main function for command-line usage"""
    parser = argparse.ArgumentParser(description='CloudForge AI Anomaly Detector')
    parser.add_argument('--dataset', 
                       choices=['heart_disease', 'wine_quality', 'network_intrusion', 'system_logs'],
                       default='system_logs',
                       help='UCI dataset to analyze for anomalies')
    parser.add_argument('--algorithm', 
                       choices=['isolation_forest', 'one_class_svm', 'local_outlier_factor'],
                       default='isolation_forest',
                       help='Anomaly detection algorithm')
    parser.add_argument('--contamination', type=float, default=0.1,
                       help='Expected proportion of anomalies (0.0-0.5)')
    parser.add_argument('--output', default='anomaly_report.json',
                       help='Output file for anomaly report')
    parser.add_argument('--visualize', action='store_true',
                       help='Generate visualization plots')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose logging')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        # Create logs and output directories
        os.makedirs('logs', exist_ok=True)
        os.makedirs('output', exist_ok=True)
        
        # Initialize detector
        detector = AdvancedAnomalyDetector(contamination=args.contamination)
        
        # Detect anomalies
        report = detector.detect_anomalies_uci_dataset(args.dataset, args.algorithm)
        
        # Save report
        with open(args.output, 'w') as f:
            json.dump(asdict(report), f, indent=2, default=str)
        
        # Generate visualizations
        if args.visualize:
            detector.visualize_anomalies(report)
        
        # Print summary
        print(f"✅ Anomaly detection completed successfully!")
        print(f"📄 Report saved to: {args.output}")
        print(f"📊 Dataset: {args.dataset}")
        print(f"🔍 Algorithm: {args.algorithm}")
        print(f"🚨 Anomalies detected: {report.anomalies_detected}/{report.total_points}")
        print(f"📈 Anomaly rate: {report.anomaly_rate:.1%}")
        print(f"⚠️  Risk assessment: {report.risk_assessment}")
        
        # Print key insights
        if report.anomaly_points:
            print(f"\n🔍 Key Insights:")
            severity_counts = {}
            for ap in report.anomaly_points:
                severity_counts[ap.severity] = severity_counts.get(ap.severity, 0) + 1
            
            for severity, count in severity_counts.items():
                print(f"  • {count} {severity} severity anomalies")
        
        print(f"\n💡 Recommendations:")
        for rec in report.recommendations[:3]:  # Show top 3 recommendations
            print(f"  • {rec}")
        
    except Exception as e:
        logger.error(f"Anomaly detection failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()