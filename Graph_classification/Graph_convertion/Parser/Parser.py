"""
Parses STEP files to extract headers and data sections.
"""
from pathlib import Path

def line_clean(line: str) -> str:
    """
    Cleans a line by removing leading/trailing spaces, newline characters, and single quotes.
    """
    line = line.strip()
    line = line.replace("\n", "")
    line = line.replace("\'", '')
    return line


def parse_file(file_name: str, dataset_path: Path) -> tuple:
    """
    Parses a STEP file and returns its headers and data sections.
    """
    full_file_name = Path(dataset_path) / file_name
    with open(full_file_name, encoding='latin-1') as f:
        data = [line_clean(line) for line in f if line.strip() and not line.lstrip().strip().startswith("/**")]
        joined_data = ''.join(data)
        datas = joined_data.split(";")
    headers = []
    start_data = 0
    for i, line in enumerate(datas):
        if line != 'DATA':
            headers.append(line)
        else:
            start_data = i + 1
            break
    for i in range(0, start_data):
        datas.pop(0)

    return headers, datas


def parse_header(file_name: str, dataset_path: Path) -> list:
    """
    Parses only the header section of a STEP file and returns it as a list of strings.
    """
    full_file_name = Path(dataset_path) / file_name
    with open(full_file_name, encoding='latin-1') as f:
        data = f.read()
        datas = data.split("\n")
    headers = []
    for i, line in enumerate(datas):
        if line != 'DATA;':
            headers.append(line)
        else:
            break
    return headers
