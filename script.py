import pandas as pd
import json
import re
from collections import defaultdict

INPUT_EXCEL = "5. G01Pш╜жщЧич║┐х╖ешЙ║хНб.xlsx"
OUTPUT_JSONL = "collected_data.jsonl"
MISSING_FILE = "missing_translations.txt"
UNITS_FILE = "units.json"

UNITS_PATTERN = re.compile(
    r'(\d+\.?\d*)\s*'
    r'('
    r'мм|см|м|км|'
    r'кг|г|т|'
    r'с|мин|ч|'
    r'Н·м|кН·м|кгс·м|'
    r'МПа|кПа|бар|'
    r'В|А|кВт|л\.с\.|'
    r'об/мин|м/с|км/ч|'
    r'°C|%|'
    r'шт|штук|штука|'
    r'mm|cm|m|km|'
    r'kg|g|t|'
    r's|min|h|'
    r'Nm|kNm|kgfm|'
    r'MPa|kPa|bar|'
    r'V|A|kW|hp|'
    r'rpm|m/s|km/h|'
    r'pcs|pc|piece|'
    r'毫米|厘米|米|千米|'
    r'千克|克|吨|'
    r'秒|分钟|小时|'
    r'牛米|千牛米|公斤米|'
    r'兆帕|千帕|巴|'
    r'伏|安|千瓦|马力|'
    r'转/分|米/秒|公里/小时|'
    r'度|百分比|'
    r'个|件|只|套|根|条|块|片|台|部|把|'
    r'pcs|pc|piece'
    r')',
    re.IGNORECASE
)


def normalize_unit(unit):
    unit_lower = unit.lower().strip()
    unit_map = {
        'мм': 'мм', '毫米': 'мм', 'mm': 'мм',
        'см': 'см', '厘米': 'см', 'cm': 'см',
        'м': 'м', '米': 'м', 'm': 'м',
        'км': 'км', '千米': 'км', 'km': 'км',
        'кг': 'кг', '千克': 'кг', 'kg': 'кг',
        'г': 'г', '克': 'г', 'g': 'г',
        'т': 'т', '吨': 'т', 't': 'т',
        'с': 'с', '秒': 'с', 's': 'с',
        'мин': 'мин', '分钟': 'мин', 'min': 'мин',
        'ч': 'ч', '小时': 'ч', 'h': 'ч',
        'н·м': 'Н·м', '牛米': 'Н·м', 'nm': 'Н·м',
        'кн·м': 'кН·м', '千牛米': 'кН·м', 'knm': 'кН·м',
        'кгс·м': 'кгс·м', '公斤米': 'кгс·м', 'kgfm': 'кгс·м',
        'мпa': 'МПа', '兆帕': 'МПа', 'mpa': 'МПа',
        'кпа': 'кПа', '千帕': 'кПа', 'kpa': 'кПа',
        'бар': 'бар', 'bar': 'бар',
        'в': 'В', '伏': 'В', 'v': 'В',
        'а': 'А', '安': 'А', 'a': 'А',
        'квт': 'кВт', '千瓦': 'кВт', 'kw': 'кВт',
        'л.с.': 'л.с.', '马力': 'л.с.', 'hp': 'л.с.',
        'об/мин': 'об/мин', '转/分': 'об/мин', 'rpm': 'об/мин',
        'м/с': 'м/с', '米/秒': 'м/с', 'm/s': 'м/с',
        'км/ч': 'км/ч', '公里/小时': 'км/ч', 'km/h': 'км/ч',
        '°c': '°C', '度': '°C', 'c': '°C',
        'шт': 'шт', 'штук': 'шт', 'штука': 'шт',
        '个': 'шт', '件': 'шт', '只': 'шт',
        '套': 'компл', '根': 'шт', '条': 'шт',
        '块': 'шт', '片': 'шт', '台': 'шт',
        '部': 'шт', '把': 'шт',
        'pcs': 'шт', 'pc': 'шт', 'piece': 'шт',
        '%': '%', '百分比': '%',
    }
    return unit_map.get(unit_lower, unit_lower)


def extract_units(text):
    if not text:
        return None
    matches = UNITS_PATTERN.findall(text)
    if not matches:
        return None
    result = []
    for val, unit in matches:
        try:
            normalized = normalize_unit(unit)
            result.append({"value": float(val), "unit_original": unit, "unit": normalized})
        except ValueError:
            continue
    return result if result else None


def clean_text(text):
    if pd.isna(text):
        return ""
    text = str(text).strip()
    text = re.sub(r'\s+', ' ', text)
    return text


def split_chinese_russian(text):
    if pd.isna(text):
        return "", ""
    text = str(text).strip()
    if not text:
        return "", ""
    if '\n' in text:
        parts = text.split('\n')
        if len(parts) == 2:
            ch = parts[0].strip()
            ru = parts[1].strip()
            if re.search(r'[А-Яа-я]', ch) and not re.search(r'[А-Яа-я]', ru):
                ch, ru = ru, ch
            return ch, ru
    if '<br>' in text:
        parts = text.split('<br>')
        if len(parts) == 2:
            ch = parts[0].strip()
            ru = parts[1].strip()
            return ch, ru
    separators = ['；', ';', '\n', '|', '  ']
    for sep in separators:
        if sep in text:
            parts = text.split(sep)
            if len(parts) == 2:
                ch = parts[0].strip()
                ru = parts[1].strip()
                if re.search(r'[А-Яа-я]', ch) and not re.search(r'[А-Яа-я]', ru):
                    ch, ru = ru, ch
                return ch, ru
    has_chinese = bool(re.search(r'[\u4e00-\u9fff]', text))
    has_russian = bool(re.search(r'[А-Яа-я]', text))
    if has_chinese and has_russian:
        match = re.search(r'([\u4e00-\u9fff].*?)([А-Яа-я].*)', text)
        if match:
            return match.group(1).strip(), match.group(2).strip()
    return text, ""


def parse_excel(file_path):
    df = pd.read_excel(file_path, header=None, dtype=str, engine='openpyxl')

    step_col = desc_col = req_col = None
    for idx, row in df.iterrows():
        row_text = ' '.join([str(x) for x in row if pd.notna(x)])
        if '序号' in row_text and ('操作描述' in row_text or '操  作  描  述' in row_text) and (
                '操作要点' in row_text or '要点及要求' in row_text):
            for col_idx, val in enumerate(row):
                if val and '序号' in str(val):
                    step_col = col_idx
                if val and ('操作描述' in str(val) or '操  作  描  述' in str(val)):
                    desc_col = col_idx
                if val and ('操作要点' in str(val) or '要点及要求' in str(val)):
                    req_col = col_idx
            if step_col is not None and desc_col is not None:
                break

    if step_col is None or desc_col is None:
        step_col = 16
        desc_col = 17
        req_col = 26

    records = []
    current_card = None
    in_card = False

    for idx, row in df.iterrows():
        row_text = ' '.join([str(x) for x in row if pd.notna(x)])
        if not row_text:
            continue

        if '总装工艺卡片' in row_text and '工序卡号' in row_text:
            if current_card and current_card.get('operation_name_ch'):
                records.append(current_card)
            current_card = {
                'card_number': '',
                'operation_name_ch': '',
                'operation_name_ru': '',
                'steps': [],
                'requirements_ch': '',
                'requirements_ru': ''
            }
            in_card = True
            card_match = re.search(r'CM-[A-Z]\d+', row_text)
            if card_match:
                current_card['card_number'] = card_match.group()
            else:
                for cell in row:
                    if cell and re.search(r'CM-[A-Z]\d+', str(cell)):
                        current_card['card_number'] = re.search(r'CM-[A-Z]\d+', str(cell)).group()
                        break
            continue

        if in_card and current_card and '工序名称' in row_text:
            ch, ru = split_chinese_russian(row_text)
            if ch and '安装' in ch:
                current_card['operation_name_ch'] = ch
            if ru and 'Установка' in ru:
                current_card['operation_name_ru'] = ru
            continue

        if in_card and current_card and not current_card['operation_name_ch']:
            if '安装' in row_text or 'Установка' in row_text:
                ch, ru = split_chinese_russian(row_text)
                if ch:
                    current_card['operation_name_ch'] = ch
                if ru:
                    current_card['operation_name_ru'] = ru

        if in_card and current_card and step_col is not None and step_col < len(row):
            step_val = row[step_col]
            if step_val and re.match(r'^\d+$', str(step_val).strip()):
                step_num = str(step_val).strip()
                desc_text = row[desc_col] if desc_col is not None and desc_col < len(row) else ''
                req_text = row[req_col] if req_col is not None and req_col < len(row) else ''

                desc_ch, desc_ru = split_chinese_russian(desc_text)
                if not desc_ch:
                    desc_ch = desc_text if not pd.isna(desc_text) else ''
                if not desc_ru:
                    desc_ru = ''

                req_ch, req_ru = split_chinese_russian(req_text)
                if not req_ch:
                    req_ch = req_text if not pd.isna(req_text) else ''
                if not req_ru:
                    req_ru = ''

                current_card['steps'].append({
                    'number': step_num,
                    'chinese': desc_ch,
                    'russian': desc_ru,
                    'requirements_ch': req_ch,
                    'requirements_ru': req_ru
                })

    if current_card and current_card.get('operation_name_ch'):
        records.append(current_card)

    return records


def process_records(records):
    collected = []
    missing = []
    all_units = defaultdict(list)
    seen_pairs = set()  # для отслеживания дубликатов

    for card in records:
        card_number = card.get('card_number', 'KC-001')

        def is_valid_translation(chinese, russian):
            if not russian or not chinese:
                return False, "пустой перевод"
            if re.search(r'[Tt][\s＝=－·]*\d+\.?\d*\s*[±]?\s*\d*\.?\d*\s*[Nn]?\.?\s*[Mm]?', chinese):
                return False, "момент затяжки (T=...N.m)"
            if not re.search(r'[А-Яа-я]', russian):
                return False, "нет русских букв"
            words = russian.split()
            if len(words) < 2:
                return False, f"слишком коротко: {len(words)} слов"
            if not re.search(r'[\d\.]', russian) and len(words) < 3:
                return False, "< 3 слов (не термин)"
            if len(russian) < len(chinese) * 0.2:
                return False, f"cлишком коротко: {len(russian)} vs {len(chinese)}"
            return True, "ok"

        if card.get('operation_name_ch') and card.get('operation_name_ru'):
            if card['operation_name_ru'] == card['operation_name_ch']:
                missing.append({'card': card_number, 'field': 'operation_name', 'original': card['operation_name_ch'],
                                'translation': card['operation_name_ru'], 'reason': 'Перевод отсутствует'})
            else:
                valid, reason = is_valid_translation(card['operation_name_ch'], card['operation_name_ru'])
                if valid:
                    units = extract_units(card['operation_name_ru'])
                    collected.append({'card_number': card_number, 'field_type': 'operation_name',
                                      'original': card['operation_name_ch'], 'translation': card['operation_name_ru'],
                                      'units': units or []})
                    if units:
                        all_units[card['operation_name_ru'][:80]].append(
                            {'card_number': card_number, 'field_type': 'operation_name', 'units': units})
                else:
                    missing.append(
                        {'card': card_number, 'field': 'operation_name', 'original': card['operation_name_ch'],
                         'translation': card['operation_name_ru'], 'reason': reason})

        for step in card.get('steps', []):
            if step.get('chinese') and '\n' in step['chinese'] and not step.get('russian'):
                missing.append({
                    'card': card_number,
                    'field': f"step_{step['number']}",
                    'original': step['chinese'],
                    'translation': 'нет перевода',
                    'reason': 'перевод не разделён (есть \\n в user)'
                })
                continue

            if step.get('chinese') and step.get('russian'):
                if step['russian'] == step['chinese']:
                    missing.append({'card': card_number, 'field': f"step_{step['number']}", 'original': step['chinese'],
                                    'translation': step['russian'], 'reason': 'Перевод отсутствует'})
                else:
                    valid, reason = is_valid_translation(step['chinese'], step['russian'])
                    if valid:
                        # === УДАЛЕНИЕ ДУБЛИКАТОВ ===
                        pair_key = (step['chinese'], step['russian'])
                        if pair_key not in seen_pairs:
                            seen_pairs.add(pair_key)
                            units = extract_units(step['russian'])
                            collected.append(
                                {'card_number': card_number, 'field_type': 'step', 'step_number': step['number'],
                                 'original': step['chinese'], 'translation': step['russian'], 'units': units or []})
                            if units:
                                all_units[step['russian'][:80]].append(
                                    {'card_number': card_number, 'field_type': 'step', 'step_number': step['number'],
                                     'units': units})
                        # ============================
                    else:
                        missing.append(
                            {'card': card_number, 'field': f"step_{step['number']}", 'original': step['chinese'],
                             'translation': step['russian'], 'reason': reason})

            if step.get('requirements_ch') and step.get('requirements_ru'):
                if step['requirements_ru'] == step['requirements_ch']:
                    missing.append({'card': card_number, 'field': f"req_step_{step['number']}",
                                    'original': step['requirements_ch'], 'translation': step['requirements_ru'],
                                    'reason': 'Перевод отсутствует'})
                else:
                    valid, reason = is_valid_translation(step['requirements_ch'], step['requirements_ru'])
                    if valid:
                        pair_key = (step['requirements_ch'], step['requirements_ru'])
                        if pair_key not in seen_pairs:
                            seen_pairs.add(pair_key)
                            units = extract_units(step['requirements_ru'])
                            collected.append({'card_number': card_number, 'field_type': 'key_requirements',
                                              'step_number': step['number'], 'original': step['requirements_ch'],
                                              'translation': step['requirements_ru'], 'units': units or []})
                            if units:
                                all_units[step['requirements_ru'][:80]].append(
                                    {'card_number': card_number, 'field_type': 'key_requirements',
                                     'step_number': step['number'], 'units': units})
                    else:
                        missing.append({'card': card_number, 'field': f"req_step_{step['number']}",
                                        'original': step['requirements_ch'], 'translation': step['requirements_ru'],
                                        'reason': reason})

    return collected, missing, all_units


def save_results(collected, missing, all_units):
    with open(OUTPUT_JSONL, 'w', encoding='utf-8') as f:
        for item in collected:
            entry = {
                "messages": [
                    {"role": "user", "content": item['original']},
                    {"role": "assistant", "content": item['translation']}
                ]
            }
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    with open(MISSING_FILE, 'w', encoding='utf-8') as f:
        if missing:
            f.write("=== ПОЛЯ БЕЗ ПЕРЕВОДА ===\n\n")
            for m in missing:
                f.write(f"Карта {m['card']}, поле {m['field']}:\n")
                f.write(f"  Оригинал (кит): {m['original']}\n")
                f.write(f"  Перевод (рус):  {m['translation']}\n")
                f.write(f"  Причина: {m['reason']}\n\n")
        else:
            f.write("Все поля переведены.\n")

    units_output = []
    for context, units_list in all_units.items():
        for item in units_list:
            units_output.append({
                "card_number": item['card_number'],
                "field_type": item['field_type'],
                "step_number": item.get('step_number', ''),
                "context": context,
                "units": item['units']
            })
    with open(UNITS_FILE, 'w', encoding='utf-8') as f:
        json.dump(units_output, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    try:
        records = parse_excel(INPUT_EXCEL)
        collected, missing, all_units = process_records(records)
        save_results(collected, missing, all_units)
    except FileNotFoundError:
        print(f"файл {INPUT_EXCEL} не найден")
    except Exception as e:
        print(f"ошибка: {e}")
        import traceback
        traceback.print_exc()