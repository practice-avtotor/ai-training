import PyPDF2
from pdfminer.high_level import extract_pages
from pdfminer.layout import LTTextContainer, LTChar, LTTextLine
from extract_text import text_extraction, get_tables_from_page, is_text_in_table


def process_pdf(pdf_path):
    with open(pdf_path, 'rb') as pdfFileObj:
        pdfReaded = PyPDF2.PdfReader(pdfFileObj)

        text_per_page = {}

        for pagenum, page in enumerate(extract_pages(pdf_path)):

            tables_bboxes = get_tables_from_page(pdf_path, pagenum)
            
            pageObj = pdfReaded.pages[pagenum]
            page_text = []
            line_format = []
            page_content = []

            page_elements = [(element.y1, element) for element in page._objs]
            page_elements.sort(key=lambda a: a[0], reverse=True)

            for i,component in enumerate(page_elements):
                element = component[1]

                if isinstance(element, LTTextContainer):
                    element_bbox = element.bbox
                    
                    if tables_bboxes and is_text_in_table(element_bbox, tables_bboxes):
                        continue
                    
                    (line_text, format_per_line) = text_extraction(element, remove_headings=True)
                    
                    if line_text is None:
                        continue
                    
                    page_text.append(line_text)
                    line_format.append(format_per_line)
                    page_content.append(line_text)

            dctkey = 'Page_'+str(pagenum)
            text_per_page[dctkey] = [page_text, line_format, page_content]

    return text_per_page


def save_text_to_file(text_per_page, output_filename):
    with open(output_filename, 'w', encoding='utf-8') as f:
        for page_key, page_data in text_per_page.items():
            page_content = page_data[2]
            if page_content:
                f.write(''.join(page_content))