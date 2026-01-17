"""
测试 03_Data_Processing 模块的功能
"""

import unittest
import tensorflow as tf
import numpy as np
import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


class TestTFDataAPI(unittest.TestCase):
    """测试tf.data API"""

    def setUp(self):
        """测试前的准备工作"""
        tf.random.set_seed(42)
        np.random.seed(42)

    def test_dataset_from_tensor_slices(self):
        """测试从张量创建数据集"""
        data = np.arange(10)
        dataset = tf.data.Dataset.from_tensor_slices(data)

        # 测试数据集元素
        elements = list(dataset.as_numpy_iterator())
        self.assertEqual(len(elements), 10)
        self.assertEqual(elements[0], 0)

    def test_dataset_take_operation(self):
        """测试take操作"""
        dataset = tf.data.Dataset.range(10)
        taken = dataset.take(3)

        elements = list(taken.as_numpy_iterator())
        self.assertEqual(len(elements), 3)
        self.assertEqual(elements, [0, 1, 2])

    def test_dataset_skip_operation(self):
        """测试skip操作"""
        dataset = tf.data.Dataset.range(10)
        skipped = dataset.skip(5)

        elements = list(skipped.as_numpy_iterator())
        self.assertEqual(len(elements), 5)
        self.assertEqual(elements[0], 5)

    def test_dataset_map_operation(self):
        """测试map操作"""
        dataset = tf.data.Dataset.range(5)
        mapped = dataset.map(lambda x: x * 2)

        elements = list(mapped.as_numpy_iterator())
        self.assertEqual(elements, [0, 2, 4, 6, 8])

    def test_dataset_filter_operation(self):
        """测试filter操作"""
        dataset = tf.data.Dataset.range(10)
        filtered = dataset.filter(lambda x: x % 2 == 0)

        elements = list(filtered.as_numpy_iterator())
        self.assertEqual(elements, [0, 2, 4, 6, 8])

    def test_dataset_batch_operation(self):
        """测试batch操作"""
        dataset = tf.data.Dataset.range(10)
        batched = dataset.batch(3)

        batches = list(batched.as_numpy_iterator())
        self.assertEqual(len(batches), 4)
        np.testing.assert_array_equal(batches[0], [0, 1, 2])

    def test_dataset_shuffle_operation(self):
        """测试shuffle操作"""
        dataset = tf.data.Dataset.range(10)
        shuffled = dataset.shuffle(buffer_size=10, seed=42)

        elements = list(shuffled.as_numpy_iterator())
        self.assertEqual(len(elements), 10)
        # 打乱后的顺序应该与原始不同
        self.assertNotEqual(elements, list(range(10)))

    def test_dataset_repeat_operation(self):
        """测试repeat操作"""
        dataset = tf.data.Dataset.range(3)
        repeated = dataset.repeat(2)

        elements = list(repeated.as_numpy_iterator())
        self.assertEqual(len(elements), 6)

    def test_dataset_with_features_and_labels(self):
        """测试特征和标签数据集"""
        features = np.random.randn(100, 4).astype(np.float32)
        labels = np.random.randint(0, 2, 100).astype(np.int32)

        dataset = tf.data.Dataset.from_tensor_slices((features, labels))

        for feat, label in dataset.take(1):
            self.assertEqual(feat.shape, (4,))
            self.assertIn(label.numpy(), [0, 1])


class TestDataPreprocessing(unittest.TestCase):
    """测试数据预处理"""

    def test_normalization(self):
        """测试归一化"""
        data = tf.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])

        # 标准化 (z-score)
        mean = tf.reduce_mean(data, axis=0)
        std = tf.math.reduce_std(data, axis=0)
        normalized = (data - mean) / std

        # 标准化后的均值应该接近0
        normalized_mean = tf.reduce_mean(normalized, axis=0)
        np.testing.assert_array_almost_equal(normalized_mean.numpy(), [0.0, 0.0], decimal=5)

    def test_min_max_scaling(self):
        """测试最小-最大缩放"""
        data = tf.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])

        min_val = tf.reduce_min(data, axis=0)
        max_val = tf.reduce_max(data, axis=0)
        scaled = (data - min_val) / (max_val - min_val)

        # 缩放后的值应该在[0, 1]范围内
        self.assertTrue(tf.reduce_all(scaled >= 0.0))
        self.assertTrue(tf.reduce_all(scaled <= 1.0))

    def test_one_hot_encoding(self):
        """测试one-hot编码"""
        labels = tf.constant([0, 1, 2, 1, 0])
        one_hot = tf.one_hot(labels, depth=3)

        self.assertEqual(one_hot.shape, (5, 3))
        # 第一个标签应该是[1, 0, 0]
        np.testing.assert_array_equal(one_hot[0].numpy(), [1, 0, 0])


class TestDataAugmentation(unittest.TestCase):
    """测试数据增强"""

    def test_random_flip(self):
        """测试随机翻转"""
        image = tf.random.normal((28, 28, 1))
        flipped = tf.image.random_flip_left_right(image)

        self.assertEqual(flipped.shape, image.shape)

    def test_random_rotation(self):
        """测试随机旋转"""
        image = tf.random.normal((1, 28, 28, 1))
        # 旋转90度
        rotated = tf.image.rot90(image[0])

        self.assertEqual(rotated.shape, (28, 28, 1))

    def test_random_brightness(self):
        """测试随机亮度调整"""
        image = tf.random.normal((28, 28, 3))
        adjusted = tf.image.random_brightness(image, max_delta=0.2)

        self.assertEqual(adjusted.shape, image.shape)


if __name__ == "__main__":
    unittest.main()
