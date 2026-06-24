import fitz
import pdfplumber
import os

def pdf_to_text(pdf_path):
    try:
        txt_path = pdf_path.replace('.pdf', '.txt')
        text = ""
        
        try:
            doc = fitz.open(pdf_path)
            for page in doc:
                text += page.get_text()
            doc.close()
        except Exception:
            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
        
        if text.strip():
            with open(txt_path, 'w', encoding='utf-8') as f:
                f.write(text)
            return txt_path
        
        return None
    except Exception as e:
        print(f"Conversion error: {e}")
        return None