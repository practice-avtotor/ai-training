import fitz
import pdfplumber
import re
import os

def clean_text(text):
    """
    Очистка текста от мусора:
    - Удаление номеров страниц
    - Удаление оглавлений
    - Удаление повторяющихся строк
    - Удаление пустых строк
    """
    if not text:
        return ""
    
    lines = text.split('\n')
    cleaned_lines = []
    
    # Паттерны для удаления
    patterns_to_remove = [
        r'^\s*\d+\s*$',  # Только номер страницы
        r'^\s*Page\s+\d+\s*$',  # Page X
        r'^\s*Стр\.\s*\d+\s*$',  # Стр. X
        r'^\s*-\s*\d+\s*-\s*$',  # - X -
        r'^\s*Table of Contents\s*$',  # Оглавление
        r'^\s*Содержание\s*$',
        r'^\s*Оглавление\s*$',
        r'^\s*Contents\s*$',
        r'^\s*Chapter\s+\d+.*$',  # Главы
        r'^\s*Глава\s+\d+.*$',
        r'^\s*Section\s+\d+.*$',
        r'^\s*Раздел\s+\d+.*$',
        r'^\s*FIGURE\s+\d+.*$',  # Подписи к рисункам
        r'^\s*Рис\.\s+\d+.*$',
        r'^\s*TABLE\s+\d+.*$',  # Подписи к таблицам
        r'^\s*Таблица\s+\d+.*$',
        r'^\s*www\..*\.\w+\s*$',  # Ссылки
        r'^\s*http.*$',
        r'^\s*©.*$',  # Копирайты
        r'^\s*All rights reserved.*$',
        r'^\s*Все права защищены.*$',
    ]
    
    for line in lines:
        line = line.strip()
        
        # Пропускаем пустые строк
        if not line:
            continue
        
        should_remove = False
        for pattern in patterns_to_remove:
            if re.match(pattern, line, re.IGNORECASE):
                should_remove = True
                break
        
        if should_remove:
            continue
        
        if re.match(r'^\s*\d+\s*$', line):
            continue
        
        if re.match(r'^\s*\d+\s*[/of]\s*\d+\s*$', line, re.IGNORECASE):
            continue
        
        cleaned_lines.append(line)
    
    cleaned_text = '\n'.join(cleaned_lines)
    
    cleaned_text = re.sub(r'\n\s*\n', '\n', cleaned_text)
    
    return cleaned_text

def extract_tables_with_pdfplumber(pdf_path):
    tables_text = []
    
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages):
                tables = page.extract_tables()
                
                for table in tables:
                    if not table or len(table) < 2:
                        continue
                    
                    table_lines = []
                    for row in table:
                        # Проверяем, что строка не пустая
                        if row and any(cell and str(cell).strip() for cell in row):
                            # Объединяем ячейки с разделителем
                            row_text = ' | '.join(str(cell).strip() if cell else '' for cell in row)
                            if row_text.strip():
                                table_lines.append(row_text)
                    
                    if table_lines:
                        tables_text.append('\n'.join(table_lines))
    except Exception as e:
        print(f"Table extraction error: {e}")
    
    return tables_text

def pdf_to_text(pdf_path):
    try:
        txt_path = pdf_path.replace('.pdf', '.txt')
        all_text = []
        
        try:
            doc = fitz.open(pdf_path)
            for page in doc:
                page_text = page.get_text()
                if page_text.strip():
                    all_text.append(page_text)
            doc.close()
        except Exception:
            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        all_text.append(page_text)
        
        full_text = '\n'.join(all_text)
        
        tables = extract_tables_with_pdfplumber(pdf_path)
        if tables:
            full_text += '\n\n' + '\n\n'.join(tables)
        
        cleaned_text = clean_text(full_text)
        
        if cleaned_text.strip():
            with open(txt_path, 'w', encoding='utf-8') as f:
                f.write(cleaned_text)
            return txt_path
        
        return None
    except Exception as e:
        print(f"Conversion error: {e}")
        return None