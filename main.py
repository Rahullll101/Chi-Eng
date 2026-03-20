from dotenv import load_dotenv
load_dotenv()
import streamlit as st
import os

os.environ["FLAGS_use_mkldnn"] = "0"

# --- CRITICAL: Windows CPU Fixes for PaddlePaddle ---
# These must be set BEFORE importing paddleocr
os.environ["FLAGS_enable_mkldnn"] = "0"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1" # Can help with stability on some CPUs

import re
import numpy as np
from PIL import Image
from pdf2image import convert_from_bytes
from paddleocr import PaddleOCR

import logging


# Import our custom translator module
from translator import translate_text

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables from .env file


# Set page config
st.set_page_config(page_title="Chinese → English Document Translator", layout="wide")

# --- OCR Model Caching ---
@st.cache_resource
def load_ocr_model():
    # use_gpu=False is important for CPU-only environments
    # enable_mkldnn=False is the primary fix for "OneDnnContext does not have the input Filter"
    # We also disable angle classifier (use_angle_cls=False) to avoid fused_conv2d issues in the cls model
    return PaddleOCR(
        use_angle_cls=False, 
        lang='ch', 
        use_gpu=False, 
        show_log=False,
        enable_mkldnn=False,
        cpu_threads=1 # Limit threads for stability
    )

# --- Text Cleaning Pipeline ---

def clean_text(text):
    """Remove empty lines and trim spaces."""
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    return '\n'.join(lines)

def format_text(text):
    """Replace Chinese punctuation with English equivalents."""
    replacements = {
        '，': ', ',
        '。': '. ',
        '：': ': ',
        '；': '; ',
        '？': '? ',
        '！': '! ',
        '（': ' (',
        '）': ') ',
        '【': ' [',
        '】': '] ',
        '—': '-',
    }
    for cn, en in replacements.items():
        text = text.replace(cn, en)
    
    # Fix common OCR errors for numbers
    text = re.sub(r'(?<=\d)I0', '10', text)
    text = re.sub(r'I0(?=\d)', '10', text)
    
    return text

def split_text(text, max_chars=3000):
    """Split text into chunks for Gemini API limits."""
    paragraphs = text.split('\n')
    chunks = []
    current_chunk = ""
    
    for p in paragraphs:
        if len(current_chunk) + len(p) < max_chars:
            current_chunk += p + "\n"
        else:
            chunks.append(current_chunk.strip())
            current_chunk = p + "\n"
    
    if current_chunk:
        chunks.append(current_chunk.strip())
    return chunks

def post_clean(text):
    """Final cleanup after translation."""
    # Remove multiple spaces
    text = re.sub(r' +', ' ', text)
    # Fix spacing around punctuation
    text = re.sub(r' ([,.;:?!])', r'\1', text)
    return text

def improve_structure(text):
    """Fix broken numbering and ensure proper spacing."""
    # Fix broken numbering like 3.\n1.\n1. -> 3.1.1
    text = re.sub(r'(\d+)\.\n(\d+)\.\n(\d+)', r'\1.\2.\3', text)
    text = re.sub(r'(\d+)\.\n(\d+)', r'\1.\2', text)
    
    # Ensure proper spacing before headings (lines starting with numbers)
    text = re.sub(r'\n(\d+\.)', r'\n\n\1', text)
    return text

def fix_ports(text):
    """Fix broken ranges like 7000-3 -> 7000-7003."""
    def port_replacer(match):
        base = match.group(1)
        suffix = match.group(2)
        if len(base) > len(suffix):
            new_port = base[:-len(suffix)] + suffix
            return f"{base}-{new_port}"
        return match.group(0)
    
    return re.sub(r'(\d{4,5})-(\d{1,3})', port_replacer, text)

def process_pipeline(raw_text):
    """
    Order: clean_text → format_text → split → translate → post_clean → 
    improve_structure → fix_ports → format_text
    """
    # 1. Clean
    text = clean_text(raw_text)
    # 2. Format
    text = format_text(text)
    # 3. Split
    chunks = split_text(text)
    # 4. Translate
    translated_chunks = [translate_text(chunk) for chunk in chunks]
    translated_text = '\n'.join(translated_chunks)
    # 5. Post Clean
    translated_text = post_clean(translated_text)
    # 6. Improve Structure
    translated_text = improve_structure(translated_text)
    # 7. Fix Ports
    translated_text = fix_ports(translated_text)
    # 8. Final Format
    translated_text = format_text(translated_text)
    
    return translated_text

# --- Main App ---

def main():
    st.title("Chinese → English Document Translator")
    st.markdown("Upload a Chinese PDF to extract text via OCR and translate it to English.")

    uploaded_file = st.file_uploader("Choose a Chinese PDF file", type="pdf")
    
    # Poppler path for Windows users
    st.sidebar.header("Windows Configuration")
    st.sidebar.info("If you are on Windows, you need Poppler to convert PDF to images.")
    
    # Hardcoded default path as requested by user
    default_poppler_path = r"C:\Users\Rahul\Downloads\poppler-25.12.0\Library\bin"
    
    poppler_path = st.sidebar.text_input(
        "Poppler 'bin' Path", 
        value=default_poppler_path,
        placeholder="e.g., C:\\poppler\\Library\\bin",
        help="The full path to the 'bin' folder inside your Poppler installation."
    )
    st.sidebar.markdown("[Download Poppler for Windows](https://github.com/oschwartz10612/poppler-windows/releases)")

    if uploaded_file is not None:
        # Load OCR model
        with st.spinner("Loading OCR model..."):
            ocr = load_ocr_model()

        # Convert PDF to images
        with st.spinner("Converting PDF pages to images..."):
            try:
                # dpi=150 for faster processing as requested
                if poppler_path:
                    images = convert_from_bytes(uploaded_file.read(), dpi=150, poppler_path=poppler_path)
                else:
                    images = convert_from_bytes(uploaded_file.read(), dpi=150)
            except Exception as e:
                st.error(f"Error converting PDF: {e}")
                st.info("If you are on Windows, you may need to specify the Poppler 'bin' folder path in the sidebar.")
                return

        st.success(f"Successfully loaded {len(images)} pages.")

        # Process each page
        for i, image in enumerate(images):
            st.subheader(f"Page {i+1}")
            
            col1, col2 = st.columns(2)
            
            with st.status(f"Processing Page {i+1}...", expanded=True) as status:
                # 1. OCR Extraction
                st.write("Extracting text using OCR...")
                # Convert PIL RGB to BGR for PaddleOCR
                img_array = np.array(image)
                if img_array.ndim == 3 and img_array.shape[2] == 3:
                    img_array = img_array[:, :, ::-1]
                
                raw_page_text = ""
                try:
                    # Removed cls=True to avoid potential fused_conv2d issues in the classifier
                    result = ocr.ocr(img_array, cls=False)
                    if result and isinstance(result, list):
                        # PaddleOCR returns a list of results (one per image)
                        # Each result is a list of [box, [text, conf]]
                        for page_result in result:
                            if page_result:
                                for line in page_result:
                                    if line and len(line) > 1:
                                        text = line[1][0]
                                        raw_page_text += text + "\n"
                except Exception as e:
                    st.warning(f"OCR failed on page {i+1}: {e}")
                    raw_page_text = ""

                # 2. Translation Pipeline
                st.write("Translating and cleaning text...")
                if raw_page_text.strip():
                    translated_page_text = process_pipeline(raw_page_text)
                else:
                    translated_page_text = "[No text detected on this page]"

                status.update(label=f"Page {i+1} Complete!", state="complete", expanded=False)

            with col1:
                st.text_area(f"Original Text (Page {i+1})", value=raw_page_text, height=300)
            
            with col2:
                st.text_area(f"Translated Text (Page {i+1})", value=translated_page_text, height=300)

if __name__ == "__main__":
    main()
