"""Download and unpack the pretrained autoPET III LesionTracer checkpoint.

Safe to rerun: a partial ZIP is resumed, a complete ZIP is not downloaded
again, and files that are already extracted are left alone.
"""

import hashlib
import http.client
import re
import shutil
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path, PurePosixPath

URL = "https://zenodo.org/records/14007247/files/autoPET-3-LesionTracer.zip?download=1"

# From https://zenodo.org/api/records/14007247 (files[0].size / .checksum).
EXPECTED_SIZE = 3_808_128_600
EXPECTED_MD5 = "566016409b0bd14770c0b57c1f2873f1"

ROOT = Path(__file__).resolve().parent
DOWNLOAD_DIR = ROOT / "checkpoints"
ZIP_PATH = DOWNLOAD_DIR / "autoPET-3-LesionTracer.zip"
EXTRACT_DIR = DOWNLOAD_DIR / "autopet3"

FOLDS = [f"fold_{i}" for i in range(5)]

# Zenodo answers 403 to a "Mozilla/5.0" user agent, so identify honestly.
USER_AGENT = "download_autopet_checkpoint.py (python-urllib)"
CHUNK = 1024 * 1024
TIMEOUT = 60  # seconds without data before a stalled connection is dropped
MAX_STALLED_ATTEMPTS = 5  # consecutive attempts that add no bytes
GB = 1024 ** 3


def zip_size():
    return ZIP_PATH.stat().st_size if ZIP_PATH.exists() else 0


def download_once():
    """One HTTP request. Appends to the partial ZIP; returns when the stream ends."""
    existing = zip_size()

    headers = {"User-Agent": USER_AGENT}
    if existing > 0:
        headers["Range"] = f"bytes={existing}-"

    request = urllib.request.Request(URL, headers=headers)

    with urllib.request.urlopen(request, timeout=TIMEOUT) as response:
        status = response.status

        if status == 206:
            content_range = response.headers.get("Content-Range", "")
            match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", content_range.strip())
            if not match:
                raise RuntimeError(f"Unreadable Content-Range: {content_range!r}")
            start, _, total = map(int, match.groups())
            if start != existing or total != EXPECTED_SIZE:
                raise RuntimeError(
                    "Server resumed from the wrong place or the remote file changed.\n"
                    f"  Content-Range: {content_range}\n"
                    f"  Local size:    {existing}\n"
                    f"  Expected size: {EXPECTED_SIZE}\n"
                    "The partial file was NOT modified."
                )
            mode = "ab"
        elif status == 200 and existing == 0:
            length = response.headers.get("Content-Length")
            if length is not None and int(length) != EXPECTED_SIZE:
                raise RuntimeError(
                    f"Remote file is {length} bytes, expected {EXPECTED_SIZE}. "
                    "The file on Zenodo changed; update EXPECTED_SIZE / EXPECTED_MD5."
                )
            mode = "wb"
        elif status == 200:
            # A full response must never be appended to a partial file.
            raise RuntimeError(
                "Server ignored the Range request and sent the whole file (HTTP 200),\n"
                "so the download cannot be resumed.\n"
                f"The partial file was NOT modified: {ZIP_PATH}\n"
                "Try again later. To restart from zero, delete that file yourself "
                "and rerun."
            )
        else:
            raise RuntimeError(f"Unexpected HTTP status {status}.")

        downloaded = existing
        last_print = 0.0

        with open(ZIP_PATH, mode) as f:
            while True:
                chunk = response.read(CHUNK)
                if not chunk:
                    break
                f.write(chunk)
                downloaded += len(chunk)

                now = time.monotonic()
                if now - last_print >= 0.5:
                    last_print = now
                    print(
                        f"\r{downloaded / GB:.2f} / {EXPECTED_SIZE / GB:.2f} GB "
                        f"({downloaded / EXPECTED_SIZE * 100:.1f}%)",
                        end="",
                        flush=True,
                    )
    print()


def download():
    """Download or resume until the ZIP has exactly EXPECTED_SIZE bytes."""
    DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)

    stalled = 0

    while True:
        existing = zip_size()

        if existing == EXPECTED_SIZE:
            return

        if existing > EXPECTED_SIZE:
            raise RuntimeError(
                f"{ZIP_PATH} is {existing} bytes, larger than the expected "
                f"{EXPECTED_SIZE}. It cannot be a valid download.\n"
                "Nothing was deleted. Delete that file yourself and rerun."
            )

        if existing > 0:
            print(
                f"Resuming from {existing / GB:.2f} GB "
                f"({EXPECTED_SIZE - existing:,} bytes left)..."
            )
        else:
            print("Downloading AutoPET III pretrained checkpoint...")
            print(f"Saving to: {ZIP_PATH}")

        error = None
        try:
            download_once()
        except urllib.error.HTTPError as e:
            if e.code == 416:
                raise RuntimeError(
                    f"Server rejected the resume offset {existing} (HTTP 416). "
                    "The partial file was NOT modified."
                ) from e
            error = e
        except (OSError, http.client.HTTPException) as e:
            # URLError, timeouts, connection resets, IncompleteRead.
            error = e

        got = zip_size()
        if got == EXPECTED_SIZE:
            return

        # Reaching here means the stream ended before the full size arrived.
        stalled = 0 if got > existing else stalled + 1
        reason = f"{type(error).__name__}: {error}" if error else "connection closed early"
        print(f"\nDownload interrupted at {got:,} / {EXPECTED_SIZE:,} bytes ({reason}).")

        if stalled >= MAX_STALLED_ATTEMPTS:
            raise RuntimeError(
                f"Download is INCOMPLETE: {got:,} / {EXPECTED_SIZE:,} bytes.\n"
                f"The partial file was kept: {ZIP_PATH}\n"
                "Run the script again and it will resume from there."
            )

        print("Partial file kept. Retrying in 10 s...")
        time.sleep(10)


def verify_zip():
    """Fail unless the ZIP is byte-identical to the file published on Zenodo."""
    size = zip_size()
    if size != EXPECTED_SIZE:
        raise RuntimeError(f"ZIP is {size} bytes, expected {EXPECTED_SIZE}.")

    print("Checking MD5 against Zenodo (takes a moment)...")
    md5 = hashlib.md5()
    with open(ZIP_PATH, "rb") as f:
        while chunk := f.read(8 * CHUNK):
            md5.update(chunk)

    if md5.hexdigest() != EXPECTED_MD5 or not zipfile.is_zipfile(ZIP_PATH):
        raise RuntimeError(
            "ZIP has the full size but is CORRUPT.\n"
            f"  MD5 expected: {EXPECTED_MD5}\n"
            f"  MD5 got:      {md5.hexdigest()}\n"
            f"Nothing was deleted. Delete {ZIP_PATH} yourself and rerun to "
            "download it again."
        )

    print("ZIP is complete and valid.")


def is_wanted(name):
    parts = PurePosixPath(name.replace("\\", "/")).parts
    if len(parts) >= 2 and parts[-1] == "checkpoint_final.pth" and parts[-2] in FOLDS:
        return True
    # dataset.json / plans.json of the model itself, not anything inside a fold.
    return (
        len(parts) >= 1
        and parts[-1] in ("dataset.json", "plans.json")
        and not any(p in FOLDS for p in parts)
    )


def find_extracted():
    """Return (model_dir, files) if all 7 required files are on disk, else None."""
    if not EXTRACT_DIR.exists():
        return None

    plans = list(EXTRACT_DIR.rglob("plans.json"))
    if len(plans) != 1:
        return None

    model_dir = plans[0].parent
    files = [model_dir / "dataset.json", model_dir / "plans.json"]
    files += [model_dir / fold / "checkpoint_final.pth" for fold in FOLDS]

    if all(f.is_file() and f.stat().st_size > 0 for f in files):
        return model_dir, files
    return None


def extract():
    print("\nOpening ZIP...")

    with zipfile.ZipFile(ZIP_PATH, "r") as zf:
        wanted = [info for info in zf.infolist() if is_wanted(info.filename)]

        if len(wanted) != 7:
            raise RuntimeError(
                f"Expected 7 required files in the ZIP, found {len(wanted)}:\n"
                + "\n".join(f"  {info.filename}" for info in wanted)
                + "\nStopping so we don't silently use the wrong model."
            )

        for info in wanted:
            parts = PurePosixPath(info.filename.replace("\\", "/")).parts
            if ".." in parts or PurePosixPath(*parts).is_absolute():
                raise RuntimeError(f"Unsafe path in ZIP: {info.filename}")

            target = EXTRACT_DIR.joinpath(*parts)

            if target.is_file() and target.stat().st_size == info.file_size:
                print(f"Already extracted: {info.filename}")
                continue

            if target.exists():
                print(f"Wrong size on disk, extracting again: {info.filename}")
            else:
                print(f"Extracting: {info.filename}")

            # Write to a .part file first so an interrupted extraction never
            # leaves a truncated file under the final name. Reading the member
            # to the end also checks its CRC.
            target.parent.mkdir(parents=True, exist_ok=True)
            part = target.with_name(target.name + ".part")
            with zf.open(info) as src, open(part, "wb") as dst:
                shutil.copyfileobj(src, dst, CHUNK)
            part.replace(target)

    print("Extraction finished.")


def verify():
    checkpoints = sorted(EXTRACT_DIR.rglob("checkpoint_final.pth"))
    dataset_json = sorted(EXTRACT_DIR.rglob("dataset.json"))
    plans_json = sorted(EXTRACT_DIR.rglob("plans.json"))

    print("\n========== VERIFICATION ==========")

    print(f"Checkpoints found: {len(checkpoints)}")
    for path in checkpoints:
        print(f"  {path.relative_to(ROOT)} ({path.stat().st_size / (1024 ** 2):.1f} MB)")

    print(f"\ndataset.json: {len(dataset_json)}")
    for path in dataset_json:
        print(f"  {path.relative_to(ROOT)} ({path.stat().st_size:,} bytes)")

    print(f"plans.json:   {len(plans_json)}")
    for path in plans_json:
        print(f"  {path.relative_to(ROOT)} ({path.stat().st_size:,} bytes)")

    if (
        find_extracted() is None
        or len(checkpoints) != 5
        or len(dataset_json) != 1
        or len(plans_json) != 1
    ):
        raise RuntimeError(
            "Extracted model is incomplete: need exactly 5 checkpoint_final.pth "
            "(fold_0..fold_4), 1 dataset.json and 1 plans.json in one model folder."
        )

    print(f"\nModel folder: {find_extracted()[0]}")
    print("OK - AutoPET III pretrained model is ready.")


def main():
    # A complete ZIP is the reference for the extracted files, so use it when
    # it is there. Without it, complete extracted files are enough.
    if zip_size() != EXPECTED_SIZE and find_extracted() is not None:
        print("Model files are already extracted. Nothing to download.")
        verify()
        return

    if zip_size() == EXPECTED_SIZE:
        print(f"ZIP already fully downloaded: {ZIP_PATH}")
    else:
        download()
        print("Download finished.")

    verify_zip()
    extract()
    verify()


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as e:
        raise SystemExit(f"\nERROR: {e}")
    except KeyboardInterrupt:
        raise SystemExit(
            "\nStopped. Any partial download was kept; rerun to resume."
        )
