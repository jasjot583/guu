import pdfplumber
import os
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss

class MedicalKBSystem:
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.kb_path = "data/medical_articles"
        self.documents = []
        self.filenames = []  # Added to track citations

    def extract_pdf_text(self, pdf_path):
        text_content = []
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                text_content.append(page.extract_text() or "")
        return "\n".join(text_content)

    def scan_for_alerts(self, text):
        """Differentiator: Automated safety flagging."""
        critical_keywords = ["critical", "abnormal", "positive", "high risk", "malignant"]
        found_alerts = [word.upper() for word in critical_keywords if word in text.lower()]
        return list(set(found_alerts))

    def build_index(self):
        files = [f for f in os.listdir(self.kb_path) if f.endswith('.txt')]
        for file in files:
            with open(os.path.join(self.kb_path, file), 'r', encoding='utf-8') as f:
                self.documents.append(f.read())
                self.filenames.append(file) # Track the source file name

        embeddings = self.model.encode(self.documents)
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(np.array(embeddings).astype('float32'))

    def query_kb(self, query_text, k=2):
        """Differentiator: Returns text + specific citations."""
        query_vector = self.model.encode([query_text]).astype('float32')
        distances, indices = self.index.search(query_vector, k)

        results = []
        for i in range(len(indices[0])):
            idx = indices[0][i]
            dist = distances[0][i]
            # Safety Check: If distance is too high, it's irrelevant
            if dist < 1.5:
                results.append({
                    "content": self.documents[idx],
                    "source_file": self.filenames[idx],
                    "confidence": f"{max(0, 100 - int(dist*10))}%"
                })
        return results