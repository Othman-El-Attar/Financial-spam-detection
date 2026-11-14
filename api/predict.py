"""FinSight SMS prediction endpoint for Vercel's Python runtime."""
from http.server import BaseHTTPRequestHandler
from pathlib import Path
import json
import logging
import re

from sklearn.linear_model import LogisticRegression

import joblib
import pandas as pd

LOGGER = logging.getLogger(__name__)
MODEL = joblib.load(Path(__file__).resolve().with_name("model.joblib"))

# Compatibility shim for the previously exported model.
for _, estimator in MODEL.steps:
    if isinstance(estimator, LogisticRegression) and not hasattr(estimator, "multi_class"):
        estimator.multi_class = "auto"

# Claimed sender IDs: exact aliases only, not proof of authentication.
KNOWN_SENDER_IDS = {
    "cib", "bank-al ahly", "bank-alahly", "ban k-alahly", "nbe",
    "banquemisr", "banque misr", "bm", "bdc", "alexbank", "alex bank",
    "qnb", "adib egypt", "adib", "vf-cash", "vodafone cash",
    "orangecash", "orange cash", "etisalat", "e& cash", "eand cash",
    "fawry", "telda", "hsbc", "emirates nbd", "nbd", "adcb",
    "arab bank", "we pay", "kuwait bank", "fab", "mashreq"
}


def sender_status(sender):
    if not sender:
        return "missing"
    compact = re.sub(r"[\s().-]", "", sender)
    if re.fullmatch(r"\+?\d{7,15}", compact):
        return "phone_number"
    normalized = " ".join(sender.casefold().split())
    return "known" if normalized in KNOWN_SENDER_IDS else "unknown"



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
            sender = body.get("sender_id", "")
            if sender is None:
                sender = ""
            if not isinstance(sms, str) or not isinstance(sender, str) or not sms.strip():
                return self.reply(400, {"error": "SMS text is required; sender ID must be text if provided."})
            if len(sms) > 4000 or len(sender) > 100:
                return self.reply(400, {"error": "Input is too long."})

            sender = sender.strip()
            status = sender_status(sender)
            if status == "phone_number":
                return self.reply(200, {
                    "label": "spam", "sender_status": status,
                    "decision_source": "sender_rule",
                    "reason": "Numeric phone-number sender flagged by the configured strict rule.",
                    "probabilities": None,
                    "notice": "Rule-based flag, not proof of fraud. No model probability was computed."
                })

            # Keep the sender field for the model; blank when not supplied.
            frame = pd.DataFrame([{"sms_text": sms.strip(), "sender_id": sender}])
            probabilities = MODEL.predict_proba(frame)[0]
            scores = {str(k): round(float(v), 6) for k, v in zip(MODEL.classes_, probabilities)}
            prediction = str(MODEL.predict(frame)[0])
            return self.reply(200, {
                "label": prediction,
                "sender_status": status,
                "decision_source": "ml_model",
                "reason": ("Known sender alias; SMS content still classified by ML." if status == "known" else "SMS classified by ML; sender is not verified."),
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
