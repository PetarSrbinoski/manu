"""Bounded, shared cache of deterministic volumes; random patches stay uncached.

NumPy memory maps let training read only its selected patch. MONAI input metadata
is retained for native-geometry inversion. Entries are local, trusted artifacts.
"""
from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
import fcntl
import hashlib
import io
import json
import logging
import os
from pathlib import Path
import platform
import shutil
import tempfile

LOGGER = logging.getLogger('lymphoma')
CACHE_VERSION = 1


class PreprocessingCache:
    def __init__(self, directory: Path, max_gib: float, min_free_gib: float = 40):
        self.directory = directory
        self.limit = int(max_gib * 1024**3)
        self.min_free = int(min_free_gib * 1024**3)
        directory.mkdir(parents=True, exist_ok=True)
        (directory / '.locks').mkdir(exist_ok=True)

    @contextmanager
    def lock(self, name):
        with (self.directory / '.locks' / name).open('a') as stream:
            fcntl.flock(stream, fcntl.LOCK_EX)
            yield

    def available_bytes(self) -> int:
        free = shutil.disk_usage(self.directory).free
        # WSL's virtual disk capacity can exceed space on its Windows host.
        if 'microsoft' in platform.release().lower() and Path('/mnt/c').is_dir():
            free = min(free, shutil.disk_usage('/mnt/c').free)
        return free

    def get(self, fingerprint: dict, prepare) -> dict:
        import numpy as np
        import torch
        from monai.transforms import FgBgToIndicesd, reset_ops_id

        signature = {'format': CACHE_VERSION, **fingerprint}
        key = hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest()
        entry = self.directory / key
        with self.lock(key):
            with self.lock('budget'):
                if (entry / 'entry.json').is_file():
                    result = self.load(entry, signature)
                    os.utime(entry, None)
                    LOGGER.info('Disk cache hit: %s', fingerprint['case'])
                    return result
            LOGGER.info('Disk cache miss: %s | loading/resampling once', fingerprint['case'])
            data = prepare()
            data = FgBgToIndicesd(keys='GT', image_key='PT', image_threshold=fingerprint['preprocessing']['background_threshold'])(data)
            reset_ops_id(data)  # Same convention as MONAI PersistentDataset.
            arrays = {}
            metadata = {}
            for name in ('CT', 'PT', 'GT'):
                value = data[name]
                arrays[name] = value.as_tensor().numpy()
                if name == 'GT':
                    arrays[name] = arrays[name].astype(np.uint8)
                metadata[name] = {'meta': deepcopy(value.meta), 'operations': deepcopy(value.applied_operations)}
            index_dtype = np.int32 if data['GT'].numel() < 2**31 else np.int64
            for name in ('GT_fg_indices', 'GT_bg_indices'):
                arrays[name] = np.asarray(data[name]).astype(index_dtype, copy=False)
            meta_buffer = io.BytesIO()
            torch.save(metadata, meta_buffer)
            metadata_bytes = meta_buffer.getvalue()
            size = sum(a.nbytes + 256 for a in arrays.values()) + len(metadata_bytes) + 4096
            with self.lock('budget'):
                if not self.make_room(size):
                    LOGGER.warning('Disk cache cannot store %s within its size/free-space budget; using uncached volume', fingerprint['case'])
                    return data
                temporary = Path(tempfile.mkdtemp(prefix='.building-', dir=self.directory))
                try:
                    for name, array in arrays.items():
                        np.save(temporary / f'{name}.npy', array, allow_pickle=False)
                    (temporary / 'metadata.pt').write_bytes(metadata_bytes)
                    (temporary / 'entry.json').write_text(json.dumps(signature, sort_keys=True))
                    temporary.replace(entry)
                finally:
                    if temporary.exists():
                        shutil.rmtree(temporary)
                LOGGER.info('Disk cache stored: %s | %.2f GiB', fingerprint['case'], size / 1024**3)
                result = self.load(entry, signature)
            return result

    def make_room(self, required: int) -> bool:
        if required > self.limit:
            return False
        entries = sorted((p for p in self.directory.iterdir()
                          if p.is_dir() and len(p.name) == 64 and (p / 'entry.json').is_file()),
                         key=lambda p: p.stat().st_mtime_ns)
        sizes = {p: sum(f.stat().st_size for f in p.iterdir() if f.is_file()) for p in entries}
        used = sum(sizes.values())
        while entries and (used + required > self.limit or self.available_bytes() - required < self.min_free):
            old = entries.pop(0)
            LOGGER.info('Evicting oldest preprocessing cache entry: %s', old.name)
            shutil.rmtree(old)
            used -= sizes[old]
        return used + required <= self.limit and self.available_bytes() - required >= self.min_free

    @staticmethod
    def load(entry: Path, signature: dict) -> dict:
        import numpy as np
        import torch
        from monai.data import MetaTensor

        if json.loads((entry / 'entry.json').read_text()) != signature:
            raise ValueError(f'Preprocessing cache identity mismatch: {entry}')
        metadata = torch.load(entry / 'metadata.pt', map_location='cpu', weights_only=False)
        result = {}
        for name in ('CT', 'PT', 'GT'):
            # Copy-on-write maps avoid eager full-volume copies and protect disk data.
            array = np.load(entry / f'{name}.npy', mmap_mode='c', allow_pickle=False)
            result[name] = MetaTensor(torch.from_numpy(array), meta=metadata[name]['meta'],
                                      applied_operations=metadata[name]['operations'])
        for name in ('GT_fg_indices', 'GT_bg_indices'):
            result[name] = np.load(entry / f'{name}.npy', mmap_mode='c', allow_pickle=False)
        return result
