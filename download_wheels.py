import os
import json
import urllib.request
import time
import socket
socket.setdefaulttimeout(60)

WHEELS_DIR = "wheels"
os.makedirs(WHEELS_DIR, exist_ok=True)

PACKAGES = {
    "scipy": "1.13.1",
    "scikit-learn": "1.6.1",
    "matplotlib": "3.8.4",
    "seaborn": "0.13.2",
    "fpdf2": "2.7.9"
}

def get_wheel_url(package, version):
    api_url = f"https://pypi.org/pypi/{package}/{version}/json"
    print(f"Querying PyPI for {package}=={version}...")
    try:
        req = urllib.request.Request(api_url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=30) as response:
            data = json.loads(response.read().decode())
            for release in data.get("urls", []):
                filename = release.get("filename", "")
                if "cp39" in filename and "win_amd64" in filename and filename.endswith(".whl"):
                    return release.get("url"), filename
            # If no specific cp39-win_amd64 found, check for generic/pure python wheel
            for release in data.get("urls", []):
                filename = release.get("filename", "")
                if "py3-none-any" in filename and filename.endswith(".whl"):
                    return release.get("url"), filename
    except Exception as e:
        print(f"Error querying PyPI for {package}: {e}")
    return None, None

def download_file(url, filepath, retries=5, delay=5):
    print(f"Downloading {url} to {filepath}...")
    for attempt in range(1, retries + 1):
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req, timeout=60) as response, open(filepath, 'wb') as out_file:
                meta = response.info()
                file_size = int(meta.get("Content-Length", 0))
                print(f"File size: {file_size / (1024*1024):.2f} MB")
                
                chunk_size = 8192
                downloaded = 0
                while True:
                    chunk = response.read(chunk_size)
                    if not chunk:
                        break
                    out_file.write(chunk)
                    downloaded += len(chunk)
                    # print progress every 5MB
                    if downloaded % (5 * 1024 * 1024) < chunk_size:
                        print(f"Downloaded {downloaded / (1024*1024):.2f} MB / {file_size / (1024*1024):.2f} MB")
            print("Download complete.")
            return True
        except Exception as e:
            print(f"Attempt {attempt} failed: {e}")
            if attempt < retries:
                print(f"Retrying in {delay} seconds...")
                time.sleep(delay)
                delay *= 2  # Exponential backoff
    return False

def main():
    downloaded_files = []
    for pkg, ver in PACKAGES.items():
        url, filename = get_wheel_url(pkg, ver)
        if url:
            filepath = os.path.join(WHEELS_DIR, filename)
            if os.path.exists(filepath):
                print(f"Wheel already exists: {filepath}")
                downloaded_files.append(filepath)
                continue
            
            success = download_file(url, filepath)
            if success:
                downloaded_files.append(filepath)
            else:
                print(f"Failed to download {pkg} after multiple attempts.")
        else:
            print(f"Could not find compatible wheel for {pkg}=={ver}")
            
    print("\nAll downloads finished. Installing wheels...")
    # Return list of files to install
    print("Files to install:", downloaded_files)

if __name__ == "__main__":
    main()
