"""Inspect the MNIST dataset and save one sample image."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from mnist_data import load_mnist


OUTPUT_DIRECTORY = Path(__file__).resolve().parent / "outputs"


def print_dataset_summary(
    train_images: np.ndarray,
    train_labels: np.ndarray,
    test_images: np.ndarray,
    test_labels: np.ndarray,
) -> None:
    """Print the shapes and basic values that describe the dataset."""

    print("\nMNIST data summary")
    print("------------------")
    print(f"Training images shape: {train_images.shape}")
    print(f"Training labels shape: {train_labels.shape}")
    print(f"Test images shape:     {test_images.shape}")
    print(f"Test labels shape:     {test_labels.shape}")
    print(f"Pixel data type:       {train_images.dtype}")
    print(f"Pixel value range:     {train_images.min()} to {train_images.max()}")
    print(f"First training label:  {int(train_labels[0])}")


def save_sample_image(image: np.ndarray, label: int) -> Path:
    """Save one image with its correct label and return the output path."""

    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIRECTORY / "mnist_sample.png"

    figure, axis = plt.subplots(figsize=(3, 3))
    axis.imshow(image, cmap="gray", vmin=0, vmax=255)
    axis.set_title(f"Correct label: {label}")
    axis.axis("off")
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)

    return output_path


def main() -> None:
    """Load MNIST, print its summary and save the first training image."""

    train_images, train_labels, test_images, test_labels = load_mnist()
    print_dataset_summary(train_images, train_labels, test_images, test_labels)

    output_path = save_sample_image(train_images[0], int(train_labels[0]))
    print(f"Sample image saved to: {output_path}")


if __name__ == "__main__":
    main()
