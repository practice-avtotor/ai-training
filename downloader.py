import requests
import os
import time
from config import OUTPUT_DIR

def download_pdf(url):
    try:
        filename = url.split('/')[-1]
        if not filename.endswith('.pdf'):
            filename = f"manual_{int(time.time())}.pdf"
        save_path = os.path.join(OUTPUT_DIR, filename)
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        }
        
        response = requests.get(url, headers=headers, stream=True, timeout=60)
        response.raise_for_status()
        
        with open(save_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        return save_path
    except Exception as e:
        print(f"Download error: {e}")
        return None