import os
from downloader import download_pdf
from converter import pdf_to_text

def process_pdf_url(url):
    pdf_path = download_pdf(url)
    if not pdf_path:
        return None
    
    txt_path = pdf_to_text(pdf_path)
    
    if txt_path and os.path.exists(pdf_path):
        os.remove(pdf_path)
    
    return txt_path