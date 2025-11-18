#!/usr/bin/env python

import os, hashlib, tarfile, urllib.request, sys
from tqdm import tqdm

TAG   = "v2.0.0"
BASE  = f"https://github.com/par-nay/lyanna/releases/download/{TAG}"

def _ensure_assets_root(user_path: str) -> str:
    """Return <user_path>/lyanna_assets, creating it if needed."""
    root = os.path.join(os.path.abspath(user_path), "lyanna_assets")
    os.makedirs(root, exist_ok=True)
    return root

def _download_and_parse_checksums(user_path: str) -> tuple[str, dict[str, str]]:
    """
    Ensure checksums.txt is present under <user_path>/lyanna_assets
    and return (assets_root, {filename -> sha256}).
    """
    assets_root = _ensure_assets_root(user_path)
    checksums_path = os.path.join(assets_root, "checksums.txt")
    url = f"{BASE}/checksums.txt"

    urllib.request.urlretrieve(url, checksums_path)

    checksums: dict[str, str] = {}
    with open(checksums_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            digest, name = line.split(maxsplit=1)
            checksums[name.strip()] = digest

    return assets_root, checksums

def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

def _download_with_progress(url, dest):
    with urllib.request.urlopen(url) as resp:
        total = int(resp.headers.get("Content-Length", 0))
        bar = tqdm(total=total, unit="B", unit_scale=True, desc=f"[lyanna] Downloading {os.path.basename(dest)}", file=sys.stdout,)

        with open(dest, "wb") as f:
            while True:
                chunk = resp.read(1024 * 32)
                if not chunk:
                    break
                f.write(chunk)
                bar.update(len(chunk))

        bar.close()

def _extract_with_progress(tar_path, target_dir):
    with tarfile.open(tar_path, "r:gz") as t:
        members = t.getmembers()
        bar = tqdm(total=len(members), desc=f"[lyanna] Extracting {os.path.basename(tar_path)}", unit = "files", file=sys.stdout,)

        for m in members:
            t.extract(m, path=target_dir)
            bar.update(1)

        bar.close()

def _download_assets(user_path: str) -> str:
    """
    Download all assets listed in checksums.txt into
    <user_path>/lyanna_assets, verifying their checksums.

    Returns the path to the assets root directory.
    """
    assets_root, checksums = _download_and_parse_checksums(user_path)

    for fname, expected_digest in checksums.items():
        url = f"{BASE}/{fname}"
        dest = os.path.join(assets_root, fname)

        # Decide whether we need to (re)download
        need_download = True
        if os.path.exists(dest):
            actual = _sha256(dest)
            if actual == expected_digest:
                need_download = False
            else:
                print(f"[lyanna] Checksum mismatch for existing {fname}, re-downloading ...")

        if need_download:
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            # print(f"[lyanna] Downloading {fname} ...")
            _download_with_progress(url, dest)
            actual = _sha256(dest)
            if actual != expected_digest:
                raise RuntimeError(
                    f"Checksum mismatch for {fname}: expected {expected_digest}, got {actual}"
                )

    return assets_root


def _extract_archives(assets_root: str) -> dict[str, str]:
    """
    Extract weight archives inside `assets_root`.

    - nsansa_weights_*.tar.gz → assets_root/nsansa_weights/
    - sansa_weights.tar.gz    → assets_root/sansa_weights/

    Tarballs are deleted after successful extraction.

    Returns a dict with the resolved directories.
    """
    nsansa_dir = os.path.join(assets_root, "nsansa_weights")
    sansa_dir = os.path.join(assets_root, "sansa_weights")
    os.makedirs(nsansa_dir, exist_ok=True)
    os.makedirs(sansa_dir, exist_ok=True)

    for fname in os.listdir(assets_root):
        if not fname.endswith(".tar.gz"):
            continue

        full_path = os.path.join(assets_root, fname)

        if fname.startswith("nsansa_weights_"):
            target_dir = nsansa_dir
        elif fname == "sansa_weights.tar.gz":
            target_dir = sansa_dir
        else:
            # unknown archive, skip or handle differently if needed
            continue

        print(f"[lyanna] Extracting {fname} ...", end = "\r")
        _extract_with_progress(full_path, target_dir)

        os.remove(full_path)

        sub = os.path.join(target_dir, "weights")
        if os.path.isdir(sub):
            for f in os.listdir(sub):
                os.rename(os.path.join(sub, f), os.path.join(target_dir, f))
            os.rmdir(sub)

    return {
        "root": assets_root,
        "nsansa_weights": nsansa_dir,
        "sansa_weights": sansa_dir,
    }


def get_assets(target_dir: str) -> dict[str, str]:
    """
    High-level entry point for users.

    - Downloads all assets into <target_dir>/lyanna_assets
    - Verifies checksums
    - Extracts the weight archives
    - Returns a dict of useful paths
    """
    assets_root = _download_assets(target_dir)
    paths = _extract_archives(assets_root)
    print("[lyanna] Assets ready.")
    return paths