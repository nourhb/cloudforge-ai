from typing import List, Dict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans


def detect_anomalies(log_lines: List[str]) -> Dict:
    if not log_lines:
        return {"clusters": [], "anomalies": []}
    # Vectorize logs
    vectorizer = TfidfVectorizer(max_features=500)
    X = vectorizer.fit_transform(log_lines)
    # Simple 2-cluster model: normal vs anomalous-like
    kmeans = KMeans(n_clusters=2, n_init=10, random_state=42)
    labels = kmeans.fit_predict(X)
    # Assume smaller cluster are anomalies
    counts = {0: (labels == 0).sum(), 1: (labels == 1).sum()}
    anomaly_label = min(counts, key=counts.get)
    anomalies = [i for i, l in enumerate(labels) if l == anomaly_label]
    return {
        "clusters": [int(c) for c in labels],
        "anomalies": anomalies[:50],  # cap output
    }

# TEST: detect_anomalies returns clusters/anomalies
