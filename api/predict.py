from http.server import BaseHTTPRequestHandler
from pathlib import Path
import json
import joblib
import pandas as pd
import traceback


MODEL = joblib.load(Path(__file__).with_name('model.joblib'))

class handler(BaseHTTPRequestHandler):
    def do_POST(self):
        try:
            size = int(self.headers.get('Content-Length', '0'))
            if size <= 0 or size > 16000:
                return self.reply(400, {'error': 'Invalid request size.'})
            body = json.loads(self.rfile.read(size))
            sms = body.get('sms_text', '')
            sender = body.get('sender_id', '')
            if not isinstance(sms, str) or not isinstance(sender, str) or not sms.strip() or not sender.strip():
                return self.reply(400, {'error': 'Both SMS text and sender ID are required.'})
            if len(sms) > 4000 or len(sender) > 100:
                return self.reply(400, {'error': 'Input is too long.'})
            frame = pd.DataFrame([{'sms_text': sms, 'sender_id': sender}])
            probabilities = MODEL.predict_proba(frame)[0]
            scores = {str(k): round(float(v), 6) for k, v in zip(MODEL.classes_, probabilities)}
            self.reply(200, {'label': str(MODEL.predict(frame)[0]), 'probabilities': scores,
                             'notice': 'Synthetic-data model output; not verification of sender authenticity.'})
        except (ValueError, TypeError, json.JSONDecodeError):
            self.reply(400, {'error': 'Invalid JSON request.'})
        
        except Exception as e:
            print("PREDICTION ERROR:", repr(e))
            traceback.print_exc()
            return jsonify({"error": str(e)}), 500

    def reply(self, status, data):
        content = json.dumps(data, ensure_ascii=False).encode('utf-8')
        self.send_response(status)
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.send_header('Cache-Control', 'no-store')
        self.send_header('Content-Length', str(len(content)))
        self.end_headers()
        self.wfile.write(content)
