# ODS API

REST API service for the ODS (Object Detection System) providing machine learning inference and model retraining capabilities.

## Features

- **Text Inference**: Classify text using trained machine learning models
- **Model Retraining**: Retrain models with new data
- **Database Integration**: Store and manage training data and model states

## API Endpoints

- `POST /infer` - Perform text classification inference
- `POST /retrain` - Retrain the model with new training data

## Development

Run the API server:

```bash
uv run fastapi dev --port 8081 src/bi_server/app.py
```

## Testing

The API includes a Bruno collection for comprehensive testing in the `bruno/` directory. Use Bruno CLI or the Bruno desktop application to run the API tests.