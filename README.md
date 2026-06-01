# Fraud Detection Dashboard

A credit-card fraud detector with a live web dashboard. It trains an **Isolation
Forest** (an anomaly-detection model) on normal spending, so anything that looks
weird — huge amount, 3am, 900km from home — gets flagged as possible fraud.

The model trains on synthetic transaction data when the server starts, so it
just runs. There's a table of scored example transactions and a form to score
your own.

## features

- Isolation Forest anomaly detection (scikit-learn)
- features: amount, hour of day, distance from home
- live scoring via a web form
- color-coded fraud / ok dashboard

## run

```bash
pip install flask scikit-learn numpy
python app.py
```

open http://127.0.0.1:5000

tags: ai, ml, finance, fraud, anomaly-detection, flask, backend

anomaly detection is smart - it learns "normal" and flags the weird stuff itself.
