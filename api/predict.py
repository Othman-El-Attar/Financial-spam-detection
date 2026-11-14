"""FinSight SMS prediction endpoint for Vercel's Python runtime."""
from http.server import BaseHTTPRequestHandler
from pathlib import Path
import json
import logging

import joblib
import pandas as pd

from sklearn.linear_model import LogisticRegression

LOGGER = logging.getLogger(__name__)
MODEL = joblib.load(Path(__file__).resolve().with_name("model.joblib"))

# Compatibility workaround for older serialized models
for _, estimator in MODEL.steps:
    if isinstance(estimator, LogisticRegression):
        if not hasattr(estimator, "multi_class"):
            estimator.multi_class = "auto"


class handler(BaseHTTPRequestHandler):
    def do_POST(self):
        try:
            size = int(self.headers.get("Content-Length", "0"))
            if not 0 < size <= 16000:
                return self.reply(400, {"error": "Invalid request size."})
            body = json.loads(self.rfile.read(size))
            if not isinstance(body, dict):
                return self.reply(400, {"error": "Expected a JSON object."})
            sms = body.get("sms_text")
            sender = body.get("sender_id")
            if not isinstance(sms, str) or not isinstance(sender, str) or not sms.strip() or not sender.strip():
                return self.reply(400, {"error": "Both SMS text and sender ID are required."})
            if len(sms) > 4000 or len(sender) > 100:
                return self.reply(400, {"error": "Input is too long."})

            frame = pd.DataFrame([{"sms_text": sms.strip(), "sender_id": sender.strip()}])
            probabilities = MODEL.predict_proba(frame)[0]
            scores = {str(k): round(float(v), 6) for k, v in zip(MODEL.classes_, probabilities)}
            prediction = str(MODEL.predict(frame)[0])
            return self.reply(200, {
                "label": prediction,
                "probabilities": scores,
                "notice": "Synthetic-data model output; not verification of sender authenticity."
            })
        except (ValueError, UnicodeDecodeError, json.JSONDecodeError):
            return self.reply(400, {"error": "Invalid JSON request."})
        except Exception:
            LOGGER.exception("FinSight prediction failed")
            return self.reply(500, {"error": "Prediction failed. Check Vercel runtime logs."})

    def do_GET(self):
        return self.reply(405, {"error": "Use POST to classify an SMS."})

    def reply(self, status, data):
        content = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)
