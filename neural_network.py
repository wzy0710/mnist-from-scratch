"""Small neural-network building blocks implemented with NumPy."""

from __future__ import annotations

import numpy as np


def linear_forward(
    inputs: np.ndarray, weights: np.ndarray, biases: np.ndarray
) -> np.ndarray:
    """Compute the output of a fully connected layer.

    ``inputs`` has shape (batch_size, input_size), ``weights`` has shape
    (input_size, output_size), and ``biases`` has shape (output_size,).
    The returned array has shape (batch_size, output_size).
    """

    return inputs @ weights + biases
