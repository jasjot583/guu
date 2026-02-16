"""
API Server
This file handles the API endpoints.
"""

from flask import Flask

app = Flask(__name__)

@app.route('/')
def home():
    return "Medical Report Backend API"

if __name__ == "__main__":
    app.run(debug=True)
