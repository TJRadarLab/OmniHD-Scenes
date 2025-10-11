import json
import os
import threading
import requests
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import unquote, urlparse
from pathlib import Path
import time

def parse_oss_url(url):
    parsed_url = urlparse(url)

    path = unquote(parsed_url.path)

    if path.startswith('/'):
        path = path[1:]

    if '?' in path:
        path = path.split('?')[0]

    return path

def download_file(item, base_dir="downloaded_files", chunk_size=8192, timeout=30):
    try:
        url = item['url']
        file_path = parse_oss_url(url)

        local_path = Path(base_dir) / file_path

        local_path.parent.mkdir(parents=True, exist_ok=True)

        response = requests.get(url, stream=True, timeout=timeout)
        response.raise_for_status()

        with open(local_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)

        print(f"✓ Download completed: {file_path} ({item['size']} bytes)")
        return True

    except Exception as e:
        print(f"✗ Failed to download file: {file_path if 'file_path' in locals() else 'unknown file'} - {str(e)}")
        return False

def download_files_multithreaded(json_file_path, max_workers=5, base_dir="downloaded_files", chunk_size=8192, timeout=30):

    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            files_data = json.load(f)
    except Exception as e:
        print(f"✗ 读取JSON文件失败: {str(e)}")
        return

    valid_files = [item for item in files_data if item.get('size', 0) > 0]

    print(f"{len(valid_files)} VALID FILES")
    print(f"Download Directory: {base_dir}")
    print(f"Max Workers: {max_workers}")
    print(f"Chunk Size: {chunk_size} bytes")
    print(f"Timeout: {timeout} seconds")
    print("-" * 50)

    start_time = time.time()
    success_count = 0
    failed_count = 0

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_file = {executor.submit(download_file, item, base_dir, chunk_size, timeout): item for item in valid_files}
        for future in as_completed(future_to_file):
            item = future_to_file[future]
            try:
                result = future.result()
                if result:
                    success_count += 1
                else:
                    failed_count += 1
            except Exception as e:
                print(f"✗ ERROR: {item.get('name', 'unknown file')} - {str(e)}")
                failed_count += 1

    elapsed_time = time.time() - start_time

    print("-" * 50)
    print(f"Done!")
    print(f"Successful downloads: {success_count}, Failed downloads: {failed_count}")
    print(f"Total time: {elapsed_time:.2f} seconds")
    print(f"Average speed: {success_count / elapsed_time:.2f} files/second")

def parse_arguments():
    parser = argparse.ArgumentParser(description='Multithreaded File Downloader from OSS URLs in JSON')

    parser.add_argument(
        '--json-file',
        default="omnihd-scenes.1011.json",
    )

    parser.add_argument(
        '--max-workers',
        type=int,
        default=20,
    )

    parser.add_argument(
        '--download-dir',
        default="downloaded_files",
    )

    parser.add_argument(
        '--chunk-size',
        type=int,
        default=8192,
    )

    parser.add_argument(
        '--timeout',
        type=int,
        default=30,
    )

    return parser.parse_args()

def main():
    args = parse_arguments()

    if not os.path.exists(args.json_file):
        print(f"✗ JSON FILE ERROR: {args.json_file}")
        return

    print(f"JSON PATH: {args.json_file}")
    print(f"MAX WORKERS: {args.max_workers}")
    print(f"DOWNLOAD DIR: {args.download_dir}")
    print(f"CHUNK SIZE: {args.chunk_size} bytes")
    print(f"TIMEOUT: {args.timeout} seconds")
    print("-" * 50)

    print("STARTING DOWNLOADING FILES...")
    download_files_multithreaded(
        args.json_file,
        args.max_workers,
        args.download_dir,
        args.chunk_size,
        args.timeout
    )

if __name__ == "__main__":
    main()