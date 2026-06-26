import PyPDF2
from pdfminer.high_level import extract_pages
from pdfminer.layout import LTTextContainer, LTChar
import pdfplumber


def text_extraction(element):
    line_text = element.get_text()
    
    line_formats = []
    for text_line in element:
        if isinstance(text_line, LTTextContainer):
            for character in text_line:
                if isinstance(character, LTChar):
                    line_formats.append(character.fontname)
                    line_formats.append(character.size)
    format_per_line = list(set(line_formats))
    
    return (line_text, format_per_line)


def get_tables_from_page(pdf_path, page_num):
    tables = []
    with pdfplumber.open(pdf_path) as pdf:
        if page_num < len(pdf.pages):
            page = pdf.pages[page_num]
            found_tables = page.find_tables()
            for table in found_tables:
                bbox = table.bbox
                page_height = page.height
                pdfminer_bbox = (
                    bbox[0],
                    page_height - bbox[3],
                    bbox[2],
                    page_height - bbox[1]
                )
                tables.append(pdfminer_bbox)
    return tables


def is_text_in_table(element_bbox, tables_bboxes):
    x0, y0, x1, y1 = element_bbox
    for table_bbox in tables_bboxes:
        tx0, ty0, tx1, ty1 = table_bbox
        if tx0 <= x0 and x1 <= tx1 and ty0 <= y0 and y1 <= ty1:
            return True
        overlap_x = max(0, min(x1, tx1) - max(x0, tx0))
        overlap_y = max(0, min(y1, ty1) - max(y0, ty0))
        element_area = (x1 - x0) * (y1 - y0)
        if element_area > 0:
            overlap_ratio = (overlap_x * overlap_y) / element_area
            if overlap_ratio > 0.5:
                return True
    return False