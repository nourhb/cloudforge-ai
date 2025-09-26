from typing import Dict

def generate_iac(request_text: str) -> Dict:
    """Deterministic minimal Kubernetes Service YAML synthesized from a prompt.
    Guarantees a valid YAML with apiVersion/kind/metadata/spec.
    """
    if not request_text:
        request_text = ''

    name = 'backend'
    if 'frontend' in request_text.lower():
        name = 'frontend'
    elif 'api' in request_text.lower():
        name = 'backend'

    # Heuristic: detect first port number in prompt; default 80
    import re
    port = 80
    m = re.search(r'\b(\d{2,5})\b', request_text)
    if m:
        try:
            p = int(m.group(1))
            if 1 <= p <= 65535:
                port = p
        except Exception:
            pass

    yaml_text = f"""
apiVersion: v1
kind: Service
metadata:
  name: {name}
  labels:
    app: cloudforge-ai
    component: {name}
spec:
  type: ClusterIP
  selector:
    app: cloudforge-ai
    component: {name}
  ports:
    - name: http
      port: {port}
      targetPort: {port}
""".strip()

    return {"yaml": yaml_text}

# TEST: generate_iac returns YAML-like text
