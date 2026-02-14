"""
DeepFace Frontend Server
Serves the HTML frontend and proxies to the DeepFace API.
"""
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

from flask import Flask, send_from_directory
from flask_cors import CORS
from deepface.api.src.app import create_app

# Create the DeepFace API app
app = create_app()
CORS(app)

FRONTEND_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "frontend")


@app.route("/ui")
@app.route("/ui/<path:path>")
def serve_frontend(path="index.html"):
    return send_from_directory(FRONTEND_DIR, path)


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  DeepFace Server Running!")
    print("  API:      http://127.0.0.1:5005/")
    print("  Frontend: http://127.0.0.1:5005/ui")
    print("=" * 60 + "\n")
    app.run(host="0.0.0.0", port=5005, debug=False)
