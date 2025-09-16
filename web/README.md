# ODS Web UI

Web interface for the ODS (Object Detection System) providing a user-friendly Streamlit application for text classification and SDG prediction.

## Features

- **Text Input**: Simple text area for user input
- **SDG Classification**: Classifies text according to Sustainable Development Goals (SDGs)
- **Visual Results**: Displays SDG results with color-coded cards and links to UN documentation

## Development

Run the web interface:

```bash
uv run streamlit run src/web_ui/app.py --server.port 8501
```

The web interface will be available at <http://localhost:8501>

## Usage

1. Enter text in the input area
2. Click "Analizar" to classify the text
3. View the corresponding SDG result with links to more information
