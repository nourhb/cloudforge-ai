from flask import Flask, jsonify, request
from werkzeug.middleware.proxy_fix import ProxyFix
import logging
from logging.handlers import RotatingFileHandler
import os

from migration_analyzer import analyze_schema
from forecasting import forecast_cpu
from anomaly_detector import detect_anomalies
from doc_generator import generate_docs

app = Flask(__name__)
app.wsgi_app = ProxyFix(app.wsgi_app)

# Logging
os.makedirs('logs', exist_ok=True)
handler = RotatingFileHandler('logs/ai-services.log', maxBytes=1_000_000, backupCount=3)
handler.setLevel(logging.INFO)
app.logger.addHandler(handler)
app.logger.setLevel(logging.INFO)

@app.get('/health')
def health():
    return jsonify({"status": "ok"})

@app.post('/ai/migration/analyze')
def ai_migration_analyze():
    try:
        payload = request.get_json(force=True)
        schema_sql = payload.get('schema_sql', '')
        result = analyze_schema(schema_sql)
        return jsonify({"ok": True, "result": result})
    except Exception as e:
        app.logger.exception('Error in migration analyze')
        return jsonify({"ok": False, "error": str(e)}), 400

@app.post('/ai/forecast/cpu')
def ai_forecast_cpu():
    try:
        payload = request.get_json(force=True)
        series = payload.get('series', [])
        steps = int(payload.get('steps', 12))
        preds = forecast_cpu(series, steps)
        return jsonify({"ok": True, "predictions": preds})
    except Exception as e:
        app.logger.exception('Error in forecast')
        return jsonify({"ok": False, "error": str(e)}), 400

@app.post('/ai/anomaly/logs')
def ai_anomaly_logs():
    try:
        payload = request.get_json(force=True)
        logs = payload.get('logs', [])
        anomalies = detect_anomalies(logs)
        return jsonify({"ok": True, "anomalies": anomalies})
    except Exception as e:
        app.logger.exception('Error in anomaly detection')
        return jsonify({"ok": False, "error": str(e)}), 400

@app.post('/ai/docs/generate')
def ai_docs_generate():
    try:
        payload = request.get_json(force=True)
        code = payload.get('code', '')
        md, pdf_path = generate_docs(code)
        return jsonify({"ok": True, "markdown": md, "pdf_path": pdf_path})
    except Exception as e:
        app.logger.exception('Error in doc generation')
        return jsonify({"ok": False, "error": str(e)}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

# TEST: Flask app serves /health and AI endpoints on Python 3.12.6
