import os
import sys
import argparse
import gzip
import tarfile
import hashlib
import time
from pathlib import Path
from typing import Iterable, Optional
from glob import glob

import requests

# ------------ Mirrors & checksums ------------
MNIST_FILES = {
    "train-images-idx3-ubyte.gz": "d53e105ee54ea40749a09fcbcd1e9432",
    "train-labels-idx1-ubyte.gz": "25c81989df183df01b3e8a0aad5dffbe",
    "t10k-images-idx3-ubyte.gz": "9fb629c4189551a2d022fa330f9573f3",
    "t10k-labels-idx1-ubyte.gz": "ec29112dd5afa0611ce80d1b7f02629c",
}

MNIST_EXPECTED = {
    # (magic, count, rows, cols, raw_size)
    "train-images-idx3-ubyte": (2051, 60000, 28, 28, 16 + 60000 * 28 * 28),
    "train-labels-idx1-ubyte": (2049, 60000, 0, 0, 8 + 60000),
    "t10k-images-idx3-ubyte": (2051, 10000, 28, 28, 16 + 10000 * 28 * 28),
    "t10k-labels-idx1-ubyte": (2049, 10000, 0, 0, 8 + 10000),
}

MNIST_MIRRORS = [
    # 清华镜像（推荐）
    "https://mirrors.tuna.tsinghua.edu.cn/git/XiaoMi/mnist_data/",
    # OSSCI（PyTorch 备用镜像）
    "https://ossci-datasets.s3.amazonaws.com/mnist/",
    # 谷歌存储（可能需外网）
    "https://storage.googleapis.com/cvdf-datasets/mnist/",
    # 官方（可能不稳定）
    "http://yann.lecun.com/exdb/mnist/",
]

CIFAR_FILE = "cifar-10-python.tar.gz"
CIFAR_MD5 = "c58f30108f718f92721af3b95e74349a"
CIFAR_MIRRORS = [
    # 清华镜像
    "https://mirrors.tuna.tsinghua.edu.cn/git/XiaoMi/cifar-10/",
    # USTC（可能需要外网）
    "https://mirrors.ustc.edu.cn/google-dataset/cifar-10/",
    # OSSCI 备用
    "https://ossci-datasets.s3.amazonaws.com/",
    # 官方（可能不稳定）
    "https://www.cs.toronto.edu/~kriz/",
]

# ------------ Helpers ------------

def md5sum(path: Path, chunk: int = 1024 * 1024) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def try_download(urls: Iterable[str], filename: str, dest: Path, md5: Optional[str] = None, timeout: int = 30) -> bool:
    dest.parent.mkdir(parents=True, exist_ok=True)
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    # if dest exists and md5 matches (when provided), skip
    if dest.exists() and md5:
        try:
            if md5sum(dest).lower() == md5.lower():
                print(f"Exists & md5 ok: {dest}")
                return True
        except Exception:
            pass
    # try mirrors with simple retry per mirror
    for base in urls:
        url = base + filename
        # unique temp file to avoid conflicts
        tmp = dest.with_suffix(dest.suffix + f".part.{os.getpid()}.{int(time.time()*1000)}")
        for attempt in range(2):
            try:
                print(f"Downloading: {url} (attempt {attempt+1})")
                with requests.get(url, stream=True, timeout=timeout, headers=headers) as r:
                    r.raise_for_status()
                    with open(tmp, "wb") as f:
                        for chunk in r.iter_content(chunk_size=1024 * 1024):
                            if chunk:
                                f.write(chunk)
                # verify md5 (if provided)
                if md5:
                    got = md5sum(tmp)
                    if got.lower() != md5.lower():
                        print(f"MD5 mismatch for {filename}: got {got}, expect {md5}.")
                        tmp.unlink(missing_ok=True)
                        raise IOError("md5 mismatch")
                # replace atomically
                if dest.exists():
                    dest.unlink(missing_ok=True)
                tmp.replace(dest)
                print(f"Saved to: {dest}")
                return True
            except Exception as e:
                print(f"Failed: {url} -> {e}")
                tmp.unlink(missing_ok=True)
                time.sleep(1.0 * (attempt + 1))
                continue
    return False


def gunzip_file(src_gz: Path, dest: Path):
    print(f"Decompressing: {src_gz} -> {dest}")
    with gzip.open(src_gz, "rb") as f_in, open(dest, "wb") as f_out:
        while True:
            data = f_in.read(1024 * 1024)
            if not data:
                break
            f_out.write(data)


def extract_tar_gz(src_tgz: Path, dest_dir: Path):
    print(f"Extracting: {src_tgz} -> {dest_dir}")
    dest_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(src_tgz, "r:gz") as tar:
        tar.extractall(path=dest_dir)

def is_gzip_file(path: Path) -> bool:
    try:
        with open(path, 'rb') as f:
            sig = f.read(2)
        return sig == b'\x1f\x8b'
    except Exception:
        return False

# ------------ Validation ------------

def validate_mnist_raw(raw_path: Path) -> bool:
    name = raw_path.name
    if name not in MNIST_EXPECTED:
        return False
    magic_exp, count_exp, rows_exp, cols_exp, size_exp = MNIST_EXPECTED[name]
    try:
        st = raw_path.stat()
        if st.st_size != size_exp:
            print(f"Invalid size for {name}: got {st.st_size}, expect {size_exp}")
            return False
        with open(raw_path, "rb") as f:
            import struct
            magic = struct.unpack(">I", f.read(4))[0]
            if magic != magic_exp:
                print(f"Invalid magic for {name}: got {magic}, expect {magic_exp}")
                return False
            num = struct.unpack(">I", f.read(4))[0]
            if num != count_exp:
                print(f"Invalid count for {name}: got {num}, expect {count_exp}")
                return False
            if rows_exp and cols_exp:
                rows = struct.unpack(">I", f.read(4))[0]
                cols = struct.unpack(">I", f.read(4))[0]
                if rows != rows_exp or cols != cols_exp:
                    print(f"Invalid dims for {name}: got {rows}x{cols}, expect {rows_exp}x{cols_exp}")
                    return False
        return True
    except Exception as e:
        print(f"Validate failed for {name}: {e}")
        return False

# ------------ MNIST ------------

def prepare_mnist(out_root: Path):
    mnist_dir = out_root / "mnist"
    mnist_dir.mkdir(parents=True, exist_ok=True)

    for gz_name, checksum in MNIST_FILES.items():
        gz_path = mnist_dir / gz_name
        raw_name = gz_name.replace(".gz", "")
        raw_path = mnist_dir / raw_name
        # if already decompressed and valid, skip; otherwise clear zero-byte leftovers
        if raw_path.exists():
            try:
                if raw_path.stat().st_size == 0:
                    raw_path.unlink(missing_ok=True)
            except Exception:
                pass
        if raw_path.exists() and validate_mnist_raw(raw_path):
            print(f"Exists & valid: {raw_path}")
            continue
        # try existing gz only if it looks like gzip
        if gz_path.exists() and is_gzip_file(gz_path):
            try:
                gunzip_file(gz_path, raw_path)
                if validate_mnist_raw(raw_path):
                    print(f"Accepted from existing gz: {raw_path}")
                    continue
                else:
                    raw_path.unlink(missing_ok=True)
            except Exception as e:
                print(f"Decompress failed for {gz_path}: {e}")
                raw_path.unlink(missing_ok=True)
        # need to download from mirrors
        prioritized = [
            "https://ossci-datasets.s3.amazonaws.com/mnist/",
            "https://storage.googleapis.com/cvdf-datasets/mnist/",
            "https://mirrors.tuna.tsinghua.edu.cn/git/XiaoMi/mnist_data/",
            "http://yann.lecun.com/exdb/mnist/",
        ]
        ok = try_download(prioritized, gz_name, gz_path, md5=None)
        if not ok or not is_gzip_file(gz_path):
            # cleanup possibly html file
            gz_path.unlink(missing_ok=True)
            raise RuntimeError(f"All mirrors failed or invalid content for MNIST file: {gz_name}")
        gunzip_file(gz_path, raw_path)
        if not validate_mnist_raw(raw_path):
            raise RuntimeError(f"Decompressed file invalid: {raw_name}")
    print(f"MNIST prepared at: {mnist_dir}")

# ------------ CIFAR-10 ------------

def prepare_cifar10(out_root: Path):
    out_root.mkdir(parents=True, exist_ok=True)
    # search for existing tgz in common locations
    candidates = [
        out_root / CIFAR_FILE,
        out_root / 'mnist' / CIFAR_FILE,
    ]
    tgz = None
    for c in candidates:
        if c.exists():
            tgz = c
            break
    if tgz is None:
        # Preferred order: official, then mirrors
        prioritized = [
            "https://www.cs.toronto.edu/~kriz/",
            "https://ossci-datasets.s3.amazonaws.com/",
            "https://mirrors.tuna.tsinghua.edu.cn/git/XiaoMi/cifar-10/",
            "https://mirrors.ustc.edu.cn/google-dataset/cifar-10/",
        ]
        dest = out_root / CIFAR_FILE
        ok = try_download(prioritized, CIFAR_FILE, dest, md5=None)
        if not ok or not is_gzip_file(dest):
            dest.unlink(missing_ok=True)
            raise RuntimeError("All mirrors failed or invalid content for CIFAR-10")
        tgz = dest
    # extract
    extract_tar_gz(tgz, out_root)
    print(f"CIFAR-10 prepared at: {out_root / 'cifar-10-batches-py'}")

# ------------ CLI ------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=str, default="data", help="output data root directory")
    parser.add_argument("--mnist", action="store_true", help="download & prepare MNIST")
    parser.add_argument("--cifar", action="store_true", help="download & prepare CIFAR-10 (python)")
    args = parser.parse_args()

    out_dir = Path(args.out)

    if args.mnist:
        prepare_mnist(out_dir)
    if args.cifar:
        prepare_cifar10(out_dir)

    if not args.mnist and not args.cifar:
        print("Nothing to do. Use --mnist and/or --cifar")

    return 0


if __name__ == "__main__":
    sys.exit(main())
