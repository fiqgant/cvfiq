"""
Helper: auto-download MediaPipe model files if not present.
"""

import os
import urllib.request


def download_model(url, dest_path):
    """
    Download file from url to dest_path with progress bar.
    Returns True if successful, False on error.
    """
    if os.path.exists(dest_path):
        return True

    filename = os.path.basename(dest_path)
    print(f"  [DOWNLOAD] {filename} not found, downloading...")
    print(f"  URL: {url}")

    try:
        def progress(count, block_size, total_size):
            if total_size <= 0:
                return
            pct = min(100, count * block_size * 100 // total_size)
            bar = '#' * (pct // 5) + '-' * (20 - pct // 5)
            print(f"\r  [{bar}] {pct}%", end='', flush=True)

        urllib.request.urlretrieve(url, dest_path, reporthook=progress)
        print()  # newline after progress bar
        size_mb = os.path.getsize(dest_path) / 1024 / 1024
        print(f"  [OK] Saved to {dest_path} ({size_mb:.1f} MB)")
        return True

    except Exception as e:
        print(f"\n  [ERROR] Download failed: {e}")
        if os.path.exists(dest_path):
            os.remove(dest_path)
        return False
