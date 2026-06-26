import requests
import os

def download_pdf(url, save_path):
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        with open(save_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        return True
    except Exception as e:
        print(f"Error downloading {url}: {e}")
        return False


def get_pdf_links_from_file(file_path):
    with open(file_path, 'r') as f:
        links = [line.strip() for line in f if line.strip()]
    return links