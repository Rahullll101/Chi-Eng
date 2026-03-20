# Chinese → English Document Translator

A Streamlit-based application for extracting text from Chinese PDF documents using OCR and translating it to English using the Gemini API.

## Features
- **PaddleOCR**: High-accuracy Chinese OCR.
- **Gemini API**: Professional technical translation.
- **PDF Processing**: Handles multi-page PDFs.
- **Optimized Pipeline**: Cleans text, fixes numbering, and handles port ranges (e.g., 7000-3 → 7000-7003).

## Local Setup (Windows)

### 1. Install Python
Ensure you have Python 3.10+ installed.

### 2. Install Poppler
`pdf2image` requires Poppler to convert PDF pages to images.
1. Download the latest release from [poppler-windows](https://github.com/oschwartz10612/poppler-windows/releases).
2. Extract the ZIP file.
3. Note the path to the `bin` folder (e.g., `C:\Users\Rahul\Downloads\poppler-25.12.0\Library\bin`).
4. You can enter this path in the app's sidebar.

### 3. Install Dependencies
Open your terminal and run:
```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables
1. Copy `.env.example` to `.env`.
2. Add your Gemini API key:
```env
GEMINI_API_KEY=your_gemini_api_key_here
```

### 5. Run the App
```bash
streamlit run main.py
```

## Troubleshooting
- **PaddleOCR CPU**: The app is configured to use `use_gpu=False` for CPU-only environments.
- **Poppler Path**: If you get a "Poppler not found" error, double-check the path in the sidebar.
- **Gemini API**: Ensure your API key is valid and has access to `gemini-2.0-flash`.
- **`ERR_BLOCKED_BY_CLIENT` for `cdn.segment.com`**: This is usually caused by an ad/privacy blocker and is non-fatal. The project disables Streamlit usage telemetry via `.streamlit/config.toml` (`browser.gatherUsageStats = false`) to prevent this request.
