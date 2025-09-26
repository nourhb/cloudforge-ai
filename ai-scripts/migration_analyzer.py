from transformers import pipeline
from typing import Dict

# Lazy singleton to avoid blocking startup
_analyzer = None

def _get_analyzer():
    global _analyzer
    if _analyzer is None:
        _analyzer = pipeline('text-generation', model='distilgpt2', framework='pt', device=-1)
    return _analyzer

PROMPT = (
    "You are a database expert. Given a SQL CREATE TABLE schema, suggest migration steps, indexes, and data type adjustments.\n"
    "Output JSON with keys: steps (list), indexes (list), risks (list).\n"
    "Schema:\n\n{schema}\n"
)

def analyze_schema(schema_sql: str) -> Dict:
    if not schema_sql:
        return {"steps": [], "indexes": [], "risks": ["empty schema"]}
    prompt = PROMPT.format(schema=schema_sql[:4000])
    out = _get_analyzer()(prompt, max_new_tokens=120, do_sample=False)[0]['generated_text']
    # naive post-process: extract bullet-like suggestions
    steps = []
    indexes = []
    risks = []
    for line in out.splitlines():
        l = line.strip('- *\u2022').strip()
        low = l.lower()
        if any(k in low for k in ['create index', 'btree', 'hash', 'idx']):
            indexes.append(l)
        elif any(k in low for k in ['risk', 'warning', 'caution']):
            risks.append(l)
        elif len(l) > 0:
            steps.append(l)
    return {
        "steps": steps[:10],
        "indexes": indexes[:10],
        "risks": risks[:10],
    }

# TEST: analyze_schema returns dict with keys
