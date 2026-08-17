"""Tests for preparing MNIST data."""

import unittest

import numpy as np

from mnist_data import prepare_images


class PrepareImagesTests(unittest.TestCase):
    """Check that MNIST images are prepared for the neural network."""

    def test_flattens_each_image(self):
        """Each two-dimensional image should become one row."""

        images = np.zeros((2, 2, 3), dtype=np.uint8)

        prepared = prepare_images(images)

        self.assertEqual(prepared.shape, (2, 6))

    def test_converts_pixels_to_float32(self):
        """Prepared pixels should use a compact floating-point type."""

        images = np.zeros((1, 2, 2), dtype=np.uint8)

        prepared = prepare_images(images)

        self.assertEqual(prepared.dtype, np.float32)

    def test_normalizes_pixel_values(self):
        """Pixel values should be scaled from 0-255 to 0.0-1.0."""

        images = np.array([[[0, 127, 255]]], dtype=np.uint8)

        prepared = prepare_images(images)

        expected = np.array([[0.0, 127 / 255.0, 1.0]], dtype=np.float32)
        np.testing.assert_allclose(prepared, expected)

    def test_does_not_modify_original_images(self):
        """Preparing images should leave the original array unchanged."""

        images = np.array([[[0, 255]]], dtype=np.uint8)
        original = images.copy()

        prepare_images(images)

        np.testing.assert_array_equal(images, original)


if __name__ == "__main__":
    unittest.main()
