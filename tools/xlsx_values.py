"""Read stored XLSX values without changing the workbook or recalculating formulas."""
from pathlib import PurePosixPath
import posixpath
import xml.etree.ElementTree as ET
import zipfile

NS = {'s': 'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
RID = '{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id'

def column_index(address):
    n = 0
    for c in address:
        if not c.isalpha(): break
        n = n * 26 + ord(c.upper()) - 64
    return n - 1

def sheets(path):
    result = {}
    with zipfile.ZipFile(path) as z:
        ss = []
        if 'xl/sharedStrings.xml' in z.namelist():
            ss = [''.join(x.itertext()) for x in ET.fromstring(z.read('xl/sharedStrings.xml'))]
        rels = {x.get('Id'): x.get('Target') for x in ET.fromstring(z.read('xl/_rels/workbook.xml.rels'))}
        for sheet in ET.fromstring(z.read('xl/workbook.xml')).find('s:sheets', NS):
            target = rels[sheet.get(RID)]
            target = target.lstrip('/') if target.startswith('/') else posixpath.normpath('xl/' + target)
            rows, errors, formulas_without_values = [], [], []
            for row in ET.fromstring(z.read(target)).findall('.//s:sheetData/s:row', NS):
                values = []
                for c in row.findall('s:c', NS):
                    idx = column_index(c.get('r'))
                    while len(values) <= idx: values.append('')
                    v = c.find('s:v', NS)
                    inline = c.find('s:is', NS)
                    value = v.text if v is not None and v.text is not None else ''.join(inline.itertext()) if inline is not None else ''
                    if c.get('t') == 's': value = ss[int(value)]
                    if c.get('t') == 'e': errors.append(c.get('r'))
                    if c.find('s:f', NS) is not None and value == '': formulas_without_values.append(c.get('r'))
                    values[idx] = value
                rows.append(values)
            result[sheet.get('name')] = {'rows': rows, 'state': sheet.get('state', 'visible'),
                                        'errors': errors, 'uncached_formulas': formulas_without_values}
    return result

def records(sheet, header_row=1):
    rows = sheet['rows']
    header = rows[header_row]
    return [dict(zip(header, row + [''] * (len(header) - len(row)))) for row in rows[header_row + 1:] if any(row)]
