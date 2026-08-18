"""Tests for the neural-network building blocks."""

import unittest

import numpy as np

from neural_network import linear_forward


class LinearForwardTests(unittest.TestCase):
    """Check the forward pass of a fully connected layer."""

    def test_multiplies_inputs_and_weights_then_adds_biases(self):
        """The layer should calculate inputs @ weights + biases."""

        inputs = np.array([[1.0, 2.0]], dtype=np.float32)
        weights = np.array(
            [[3.0, 4.0], [5.0, 6.0]],
            dtype=np.float32,
        )
        biases = np.array([0.5, -0.5], dtype=np.float32)

        outputs = linear_forward(inputs, weights, biases)

        expected = np.array([[13.5, 15.5]], dtype=np.float32)
        np.testing.assert_allclose(outputs, expected)

    def test_processes_a_batch_of_inputs(self):
        """Each input row should produce one output row."""

        inputs = np.array(
            [[1.0, 0.0, 1.0], [0.0, 1.0, 1.0]],
            dtype=np.float32,
        )
        weights = np.array(
            [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
            dtype=np.float32,
        )
        biases = np.array([0.0, 1.0], dtype=np.float32)

        outputs = linear_forward(inputs, weights, biases)

        expected = np.array([[6.0, 9.0], [8.0, 11.0]], dtype=np.float32)
        np.testing.assert_allclose(outputs, expected)
        self.assertEqual(outputs.shape, (2, 2))


if __name__ == "__main__":
    unittest.main()
