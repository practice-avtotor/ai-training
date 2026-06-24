import sys
import os
from processor import process_pdf_url

def main():
    if len(sys.argv) < 2:
        print("Usage: python main.py links.txt")
        sys.exit(1)
    
    file_path = sys.argv[1]
    
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        sys.exit(1)
    
    with open(file_path, 'r') as f:
        urls = [line.strip() for line in f if line.strip()]
    
    print(f"Found {len(urls)} URLs to process")
    
    success = 0
    failed = 0
    
    for i, url in enumerate(urls, 1):
        print(f"[{i}/{len(urls)}] Processing: {url[:80]}...")
        result = process_pdf_url(url)
        if result:
            print(f"  OK: {result}")
            success += 1
        else:
            print(f"  FAIL: {url[:80]}...")
            failed += 1
    
    print(f"\nCompleted: {success} OK, {failed} FAILED")

if __name__ == "__main__":
    main()