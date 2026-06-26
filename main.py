import os
import tempfile
import time
from downloader import get_pdf_links_from_file, download_pdf
from pdf_processor import process_pdf, save_text_to_file


def get_safe_filename(url, index):
    filename = url.split('/')[-1]
    if not filename or not filename.endswith('.pdf'):
        filename = f'pdf_{index}.pdf'
    
    invalid_chars = '<>:"/\\|?*'
    for char in invalid_chars:
        filename = filename.replace(char, '_')
    
    filename = filename.replace('.pdf', '')
    
    return filename


def process_pdf_from_url(url, index, output_folder):
    temp_pdf_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp_file:
            temp_pdf_path = tmp_file.name
        
        if download_pdf(url, temp_pdf_path):
            try:
                text_per_page = process_pdf(temp_pdf_path)
                
                safe_filename = get_safe_filename(url, index)
                output_file = os.path.join(output_folder, f'{safe_filename}.txt')
                
                save_text_to_file(text_per_page, output_file)
                
                print(f"Successfully processed PDF {index} -> saved to {output_file}")
                return True
                
            except Exception as e:
                print(f"Error processing PDF {index}: {e}")
                return False
            finally:
                if temp_pdf_path and os.path.exists(temp_pdf_path):
                    try:
                        time.sleep(0.3)
                        os.remove(temp_pdf_path)
                    except PermissionError:
                        time.sleep(2)
                        try:
                            os.remove(temp_pdf_path)
                        except PermissionError:
                            print(f"Warning: Could not remove temporary file: {temp_pdf_path}")
        else:
            print(f"Failed to download PDF {index}")
            return False
            
    except Exception as e:
        print(f"Error with PDF {index}: {e}")
        if temp_pdf_path and os.path.exists(temp_pdf_path):
            try:
                os.remove(temp_pdf_path)
            except:
                pass
        return False


def main():
    links_file = 'links.txt'
    
    output_folder = 'extracted_texts'
    
    if not os.path.exists(links_file):
        print(f"Error: {links_file} not found!")
        return
    
    pdf_links = get_pdf_links_from_file(links_file)
    print(f"Found {len(pdf_links)} PDF links")
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    successful = 0
    for idx, url in enumerate(pdf_links, 1):
        print(f"\nProcessing PDF {idx}/{len(pdf_links)}: {url}")
        
        if process_pdf_from_url(url, idx, output_folder):
            successful += 1
    
    print(f"\n{'='*50}")
    print(f"Processing complete!")
    print(f"Successfully processed: {successful}/{len(pdf_links)}")
    print(f"Results saved in: {output_folder}")


if __name__ == "__main__":
    main()