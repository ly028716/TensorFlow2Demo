"""
测试 01_Basics 模块的功能
"""

import unittest
import tensorflow as tf
import numpy as np
import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


class TestTensorFlowBasics(unittest.TestCase):
    """测试TensorFlow基础功能"""

    def setUp(self):
        """测试前的准备工作"""
        tf.random.set_seed(42)
        np.random.seed(42)

    def test_tensor_creation(self):
        """测试张量创建"""
        # 测试标量
        scalar = tf.constant(42)
        self.assertEqual(scalar.ndim, 0)
        self.assertEqual(scalar.numpy(), 42)

        # 测试向量
        vector = tf.constant([1, 2, 3])
        self.assertEqual(vector.ndim, 1)
        self.assertEqual(vector.shape[0], 3)

        # 测试矩阵
        matrix = tf.constant([[1, 2], [3, 4]])
        self.assertEqual(matrix.ndim, 2)
        self.assertEqual(matrix.shape, (2, 2))

    def test_tensor_operations(self):
        """测试张量运算"""
        a = tf.constant([[1, 2], [3, 4]], dtype=tf.float32)
        b = tf.constant([[5, 6], [7, 8]], dtype=tf.float32)

        # 测试加法
        c = tf.add(a, b)
        expected = np.array([[6, 8], [10, 12]], dtype=np.float32)
        np.testing.assert_array_equal(c.numpy(), expected)

        # 测试矩阵乘法
        d = tf.matmul(a, b)
        expected = np.array([[19, 22], [43, 50]], dtype=np.float32)
        np.testing.assert_array_equal(d.numpy(), expected)

    def test_tensor_shape_operations(self):
        """测试张量形状操作"""
        x = tf.constant([[1, 2, 3, 4], [5, 6, 7, 8]])

        # 测试reshape
        reshaped = tf.reshape(x, (4, 2))
        self.assertEqual(reshaped.shape, (4, 2))

        # 测试transpose
        transposed = tf.transpose(x)
        self.assertEqual(transposed.shape, (4, 2))

    def test_variable_creation(self):
        """测试变量创建和操作"""
        var = tf.Variable([1.0, 2.0, 3.0])
        self.assertIsInstance(var, tf.Variable)
        self.assertEqual(var.shape, (3,))

        # 测试变量赋值
        var.assign([4.0, 5.0, 6.0])
        np.testing.assert_array_equal(var.numpy(), [4.0, 5.0, 6.0])

    def test_gradient_computation(self):
        """测试自动微分"""
        x = tf.Variable(3.0)

        with tf.GradientTape() as tape:
            y = x**2

        # dy/dx = 2x = 6
        dy_dx = tape.gradient(y, x)
        self.assertAlmostEqual(dy_dx.numpy(), 6.0, places=5)

    def test_numpy_conversion(self):
        """测试NumPy与TensorFlow的转换"""
        # NumPy到TensorFlow
        numpy_array = np.array([[1, 2], [3, 4]])
        tensor = tf.constant(numpy_array)
        self.assertIsInstance(tensor, tf.Tensor)

        # TensorFlow到NumPy
        back_to_numpy = tensor.numpy()
        self.assertIsInstance(back_to_numpy, np.ndarray)
        np.testing.assert_array_equal(back_to_numpy, numpy_array)

    def test_tf_function_decorator(self):
        """测试tf.function装饰器"""

        @tf.function
        def simple_function(x):
            return tf.reduce_sum(x**2)

        x = tf.constant([1, 2, 3, 4])
        result = simple_function(x)
        expected = 1 + 4 + 9 + 16
        self.assertEqual(result.numpy(), expected)


class TestTensorStatistics(unittest.TestCase):
    """测试张量统计操作"""

    def test_reduce_operations(self):
        """测试归约操作"""
        tensor = tf.constant([[1, 2, 3], [4, 5, 6]], dtype=tf.float32)

        # 测试求和
        total_sum = tf.reduce_sum(tensor)
        self.assertAlmostEqual(total_sum.numpy(), 21.0, places=5)

        # 测试按行求和
        row_sum = tf.reduce_sum(tensor, axis=1)
        np.testing.assert_array_equal(row_sum.numpy(), [6.0, 15.0])

        # 测试平均值
        mean = tf.reduce_mean(tensor)
        self.assertAlmostEqual(mean.numpy(), 3.5, places=5)

        # 测试最大值
        max_val = tf.reduce_max(tensor)
        self.assertAlmostEqual(max_val.numpy(), 6.0, places=5)

        # 测试最小值
        min_val = tf.reduce_min(tensor)
        self.assertAlmostEqual(min_val.numpy(), 1.0, places=5)


if __name__ == "__main__":
    unittest.main()
