# FinSight Financial SMS Classifier — Vercel

A deployable frontend and Python serverless API using the **existing trained model** (`api/model.joblib`). The model classifies Arabic/English deposit and withdrawal SMS as `safe` or `spam` using SMS text and sender ID.

## Deploy on Vercel

1. Extract this ZIP and upload its contents to a GitHub repository.
2. In Vercel, select **Add New → Project**, import the repository, and choose **Other** as framework preset.
3. Leave the root directory as the repository root, leave the build command empty, and deploy.
4. Visit the deployed URL. The frontend calls `/api/predict` on the same origin.

`requirements.txt` is at the repository root for the Vercel Python function.

## Run locally

Install Python dependencies: `pip install -r requirements.txt`

For the closest local simulation install Vercel CLI (`npm i -g vercel`) then run `vercel dev` and open the displayed local URL.

## API

`POST /api/predict` with JSON:

```json
{"sender_id":"CIB","sms_text":"Your account was debited by EGP 250."}
```

Returns `label` and `probabilities`.

## Limitations

- This is the previously trained model, not a new retraining.
- Training data is synthetic and does not establish real-world fraud accuracy.
- Sender ID is an input feature, **not** authenticated sender identity.
- Do not use to automatically trust or store transactions without additional checks.
- done
