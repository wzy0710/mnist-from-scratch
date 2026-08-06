"""Download and read the original MNIST files with Python and NumPy."""

from __future__ import annotations

import gzip
import struct
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np


DATA_DIRECTORY = Path(__file__).resolve().parent / "data" / "mnist"

MNIST_FILES = {
    "train_images": "train-images-idx3-ubyte.gz",
    "train_labels": "train-labels-idx1-ubyte.gz",
    "test_images": "t10k-images-idx3-ubyte.gz",
    "test_labels": "t10k-labels-idx1-ubyte.gz",
}

# Try more than one mirror so a temporary network problem does not stop the project.
MNIST_MIRRORS = (
    "https://storage.googleapis.com/cvdf-datasets/mnist",
    "https://ossci-datasets.s3.amazonaws.com/mnist",
    "https://raw.githubusercontent.com/fgnt/mnist/master",
)


def download_mnist(data_directory: Path = DATA_DIRECTORY) -> dict[str, Path]:
    """Download the four compressed MNIST files and return their local paths."""

    data_directory.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}

    for name, filename in MNIST_FILES.items():
        destination = data_directory / filename
        paths[name] = destination

        if destination.exists():
            print(f"Using existing file: {destination}")
            continue

        last_error: Exception | None = None
        temporary_file = destination.with_name(f"{destination.name}.part")

        for mirror in MNIST_MIRRORS:
            url = f"{mirror}/{filename}"
            try:
                print(f"Downloading: {url}")
                urllib.request.urlretrieve(url, temporary_file)
                temporary_file.replace(destination)
                break
            except (OSError, urllib.error.URLError) as error:
                last_error = error
                temporary_file.unlink(missing_ok=True)
                print(f"Download failed, trying another mirror: {error}")
        else:
            raise RuntimeError(f"Could not download {filename}") from last_error

    return paths


def read_images(path: Path) -> np.ndarray:
    """Read an IDX image file into an array shaped (image count, rows, columns)."""

    with gzip.open(path, "rb") as file:
        header = file.read(16)
        if len(header) != 16:
            raise ValueError(f"Image file header is incomplete: {path}")

        magic_number, image_count, row_count, column_count = struct.unpack(
            ">IIII", header
        )
        if magic_number != 2051:
            raise ValueError(f"Unexpected image magic number in {path}: {magic_number}")

        pixels = np.frombuffer(file.read(), dtype=np.uint8)

    expected_pixel_count = image_count * row_count * column_count
    if pixels.size != expected_pixel_count:
        raise ValueError(
            f"Expected {expected_pixel_count} pixels in {path}, found {pixels.size}"
        )

    return pixels.reshape(image_count, row_count, column_count)


def read_labels(path: Path) -> np.ndarray:
    """Read an IDX label file into a one-dimensional array."""

    with gzip.open(path, "rb") as file:
        header = file.read(8)
        if len(header) != 8:
            raise ValueError(f"Label file header is incomplete: {path}")

        magic_number, label_count = struct.unpack(">II", header)
        if magic_number != 2049:
            raise ValueError(f"Unexpected label magic number in {path}: {magic_number}")

        labels = np.frombuffer(file.read(), dtype=np.uint8)

    if labels.size != label_count:
        raise ValueError(f"Expected {label_count} labels in {path}, found {labels.size}")

    return labels


def load_mnist(
    data_directory: Path = DATA_DIRECTORY,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Download and return training images, training labels, test images and labels."""

    paths = download_mnist(data_directory)
    train_images = read_images(paths["train_images"])
    train_labels = read_labels(paths["train_labels"])
    test_images = read_images(paths["test_images"])
    test_labels = read_labels(paths["test_labels"])

    if train_images.shape[0] != train_labels.shape[0]:
        raise ValueError("Training image count does not match training label count")
    if test_images.shape[0] != test_labels.shape[0]:
        raise ValueError("Test image count does not match test label count")

    return train_images, train_labels, test_images, test_labels
