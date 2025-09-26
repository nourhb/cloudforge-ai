from flask import Flask, jsonify, request
from werkzeug.middleware.proxy_fix import ProxyFix
import logging
from logging.handlers import RotatingFileHandler
import os

from migration_analyzer import analyze_schema
from forecasting import forecast_cpu
from anomaly_detector import detect_anomalies
from doc_generator import generate_docs
from iac_generator import generate_iac

app = Flask(__name__)
app.wsgi_app = ProxyFix(app.wsgi_app)

# Logging
os.makedirs('logs', exist_ok=True)
handler = RotatingFileHandler('logs/ai-services.log', maxBytes=1_000_000, backupCount=3)
handler.setLevel(logging.INFO)
app.logger.addHandler(handler)
app.logger.setLevel(logging.INFO)

# Simple metrics
_start_time = __import__('time').time()
_request_count = 0

@app.before_request
def _inc_requests():
    global _request_count
    _request_count += 1

@app.get('/health')
def health():
    return jsonify({
        "status": "ok",
        "service": "ai-scripts",
        "version": "1.0.0",
        "port": int(os.getenv('AI_PORT', '5001')),
    })

@app.route('/', methods=['GET'])
def index():
    return jsonify({
        "service": "ai-scripts",
        "status": "ok",
        "endpoints": {
            "health": "/health",
            "metrics": "/metrics",
            "iac_generate": "/ai/iac/generate",
        }
    })

@app.route('/home', methods=['GET'])
def home():
    return index()

@app.get('/metrics')
def metrics():
    # Minimal Prometheus exposition format for AI service
    now = __import__('time').time()
    uptime = int(now - _start_time)
    lines = []
    lines.append('# HELP ai_up 1 if the AI service is up')
    lines.append('# TYPE ai_up gauge')
    lines.append('ai_up 1')
    lines.append('# HELP ai_requests_total Total number of HTTP requests received')
    lines.append('# TYPE ai_requests_total counter')
    lines.append(f'ai_requests_total {_request_count}')
    lines.append('# HELP ai_uptime_seconds Service uptime in seconds')
    lines.append('# TYPE ai_uptime_seconds gauge')
    lines.append(f'ai_uptime_seconds {uptime}')
    return ('\n'.join(lines) + '\n', 200, {'Content-Type': 'text/plain; version=0.0.4'})

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

@app.get('/whoami')
def whoami():
    try:
        routes = sorted([str(r) for r in app.url_map.iter_rules()])
    except Exception:
        routes = []
    return jsonify({
        "ok": True,
        "cwd": os.getcwd(),
        "file": __file__,
        "routes": routes,
    })

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

@app.post('/ai/iac/generate')
def ai_iac_generate():
    try:
        payload = request.get_json(force=True)
        prompt = payload.get('prompt', '')
        result = generate_iac(prompt)
        return jsonify({"ok": True, **result})
    except Exception as e:
        app.logger.exception('Error in IaC generation')
        return jsonify({"ok": False, "error": str(e)}), 400

if __name__ == '__main__':
    # Log registered routes for debugging
    try:
        app.logger.info("Registered routes:\n" + "\n".join(sorted([str(r) for r in app.url_map.iter_rules()])))
    except Exception:
        pass
    port = int(os.getenv('AI_PORT', '5001'))
    app.run(host='0.0.0.0', port=port, debug=False, use_reloader=False)

# TEST: Flask app serves /health and AI endpoints on Python 3.12.6
