import csv
import io
import json
from pathlib import Path
import struct
from urllib.request import Request, urlopen
from zipfile import ZipFile, ZIP_DEFLATED, ZIP_STORED
import zlib

METADATA = 'https://www.cancerimagingarchive.net/wp-content/uploads/Clinical-Metadata-FDG-PET_CT-Lesions.csv'
RECORD = 'https://fdat.uni-tuebingen.de/api/records/8f14a-pf846'
OUTPUT = Path(__file__).resolve().parent / 'data/autopet-v2-lymphoma'
FILES = {'CTres.nii.gz', 'SUV.nii.gz', 'SEG.nii.gz'}


def byte_range(url, start, length, total):
    request = Request(url, headers={'Range': f'bytes={start}-{start + length - 1}',
                                    'Accept-Encoding': 'identity'})
    response = urlopen(request, timeout=60)
    if response.status != 206 or response.headers.get('Content-Range') != f'bytes {start}-{start + length - 1}/{total}':
        response.close()
        raise RuntimeError('Server ignored the requested byte range')
    return response


  # zip index 
class RemoteIndex:
    def __init__(self, url, size):
        self.url, self.size, self.position = url, size, 0

    def seek(self, offset, whence=0):
        self.position = (0, self.position, self.size)[whence] + offset
        return self.position

    def tell(self):
        return self.position

    def read(self, size=-1):
        size = self.size - self.position if size < 0 else min(size, self.size - self.position)
        if size == 0:
            return b''
        if not 0 < size <= 4 * 1024**2:
            raise ValueError('Unexpected ZIP index size')
        with byte_range(self.url, self.position, size, self.size) as response:
            data = response.read(size)
        if len(data) != size:
            raise OSError('Incomplete ZIP index')
        self.position += size
        return data


def download(url, total, info, target):
    if target.exists() and target.stat().st_size == info.file_size:
        return
    if info.compress_type not in (ZIP_STORED, ZIP_DEFLATED):
        raise ValueError('Unsupported ZIP compression')
    with byte_range(url, info.header_offset, 30, total) as response:
        name_length, extra_length = struct.unpack_from('<HH', response.read(30), 26)
    start = info.header_offset + 30 + name_length + extra_length
    decoder = zlib.decompressobj(-15) if info.compress_type == ZIP_DEFLATED else None
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(target.name + '.part')
    size, crc = 0, 0
    with byte_range(url, start, info.compress_size, total) as response, partial.open('wb') as output:
        remaining = info.compress_size
        while remaining:
            block = response.read(min(1024**2, remaining))
            if not block:
                raise OSError('Download interrupted; rerun to resume')
            remaining -= len(block)
            data = decoder.decompress(block) if decoder else block
            output.write(data)
            size += len(data)
            crc = zlib.crc32(data, crc)
    if size != info.file_size or crc != info.CRC:
        raise OSError(f'Incomplete or corrupt file: {target}')
    partial.replace(target)


def main():
    with urlopen(METADATA, timeout=60) as response:
        rows = csv.DictReader(io.TextIOWrapper(response, encoding='utf-8-sig'))
        cases = {(r['Subject ID'], r['File Location'].split('/')[-2])
                 for r in rows if r['diagnosis'] == 'LYMPHOMA'}
    with urlopen(RECORD, timeout=60) as response:
        archive = next(iter(json.load(response)['files']['entries'].values()))
    url, total = archive['links']['content'], archive['size']
    with ZipFile(RemoteIndex(url, total)) as archive:
        entries = archive.infolist()
    print(f'{len(cases)} examinations from {len({patient for patient, _ in cases})} patients')
    for info in entries:
        parts = info.filename.split('/')
        if len(parts) != 4 or tuple(parts[1:3]) not in cases or parts[3] not in FILES:
            continue
        if any(p in ('', '.', '..') or '\\' in p for p in parts):
            raise ValueError('Unsafe archive path')
        target = OUTPUT.joinpath(*parts[1:])
        download(url, total, info, target)
        print(target.relative_to(OUTPUT), flush=True)


if __name__ == '__main__':
    main()
