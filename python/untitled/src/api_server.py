from flask import Flask, request, jsonify
from flask_cors import CORS
from medical_kb_system import MedicalKBSystem
import os

app = Flask(__name__)
CORS(app)
system = MedicalKBSystem()

# Ensure data folders exist
os.makedirs("data/uploads", exist_ok=True)

# Build index if data exists
if os.path.exists("data/medical_articles") and len(os.listdir("data/medical_articles")) > 0:
    system.build_index()

@app.route('/analyze-report', methods=['POST'])
def analyze_report():
    if 'file' not in request.files:
        return jsonify({"error": "No file"}), 400

    file = request.files['file']
    path = os.path.join("data/uploads", file.filename)
    file.save(path)

    # 1. Extraction
    raw_text = system.extract_pdf_text(path)

    # 2. Critical Alert Scanning (Differentiation!)
    alerts = system.scan_for_alerts(raw_text)

    # 3. Grounded Retrieval (Citations!)
    knowledge_hits = system.query_kb(raw_text)

    return jsonify({
        "status": "success",
        "analysis": {
            "extracted_data": raw_text[:500] + "...", # Summarized for preview
            "critical_flags": alerts,
            "references": knowledge_hits,
            "engine": "RAG-Verified-Truth-System"
        }
    })

if __name__ == "__main__":
    app.run(port=5000, debug=True)