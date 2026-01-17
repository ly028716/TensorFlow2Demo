"""
测试 02_Neural_Networks 模块的功能
"""

import unittest
import tensorflow as tf
import numpy as np
import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


class TestKerasModels(unittest.TestCase):
    """测试Keras模型构建"""

    def setUp(self):
        """测试前的准备工作"""
        tf.random.set_seed(42)
        np.random.seed(42)

    def test_sequential_model(self):
        """测试Sequential模型"""
        model = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(32, activation="relu", input_shape=(10,)),
                tf.keras.layers.Dense(16, activation="relu"),
                tf.keras.layers.Dense(1, activation="sigmoid"),
            ]
        )

        # 测试模型输出形状
        sample_input = tf.random.normal((2, 10))
        output = model(sample_input)
        self.assertEqual(output.shape, (2, 1))

    def test_functional_model(self):
        """测试Functional API模型"""
        inputs = tf.keras.Input(shape=(10,))
        x = tf.keras.layers.Dense(32, activation="relu")(inputs)
        x = tf.keras.layers.Dense(16, activation="relu")(x)
        outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)

        model = tf.keras.Model(inputs=inputs, outputs=outputs)

        # 测试模型输出形状
        sample_input = tf.random.normal((2, 10))
        output = model(sample_input)
        self.assertEqual(output.shape, (2, 1))

    def test_model_subclassing(self):
        """测试模型子类化"""

        class CustomModel(tf.keras.Model):
            def __init__(self):
                super(CustomModel, self).__init__()
                self.dense1 = tf.keras.layers.Dense(32, activation="relu")
                self.dense2 = tf.keras.layers.Dense(16, activation="relu")
                self.dense3 = tf.keras.layers.Dense(1, activation="sigmoid")

            def call(self, inputs):
                x = self.dense1(inputs)
                x = self.dense2(x)
                return self.dense3(x)

        model = CustomModel()
        sample_input = tf.random.normal((2, 10))
        output = model(sample_input)
        self.assertEqual(output.shape, (2, 1))


class TestLayers(unittest.TestCase):
    """测试神经网络层"""

    def test_dense_layer(self):
        """测试全连接层"""
        layer = tf.keras.layers.Dense(5, activation="relu")
        input_data = tf.random.normal((2, 10))
        output = layer(input_data)
        self.assertEqual(output.shape, (2, 5))

    def test_conv2d_layer(self):
        """测试2D卷积层"""
        layer = tf.keras.layers.Conv2D(32, (3, 3), activation="relu")
        input_data = tf.random.normal((2, 28, 28, 1))
        output = layer(input_data)
        self.assertEqual(output.shape, (2, 26, 26, 32))

    def test_dropout_layer(self):
        """测试Dropout层"""
        layer = tf.keras.layers.Dropout(0.5)
        input_data = tf.random.normal((2, 10))

        # 训练模式
        output_train = layer(input_data, training=True)
        self.assertEqual(output_train.shape, (2, 10))

        # 推理模式
        output_test = layer(input_data, training=False)
        self.assertEqual(output_test.shape, (2, 10))

    def test_batch_normalization(self):
        """测试批归一化层"""
        layer = tf.keras.layers.BatchNormalization()
        input_data = tf.random.normal((2, 10))
        output = layer(input_data, training=True)
        self.assertEqual(output.shape, (2, 10))


class TestActivationFunctions(unittest.TestCase):
    """测试激活函数"""

    def test_relu_activation(self):
        """测试ReLU激活函数"""
        x = tf.constant([-2.0, -1.0, 0.0, 1.0, 2.0])
        output = tf.nn.relu(x)
        expected = [0.0, 0.0, 0.0, 1.0, 2.0]
        np.testing.assert_array_equal(output.numpy(), expected)

    def test_sigmoid_activation(self):
        """测试Sigmoid激活函数"""
        x = tf.constant([0.0])
        output = tf.nn.sigmoid(x)
        self.assertAlmostEqual(output.numpy()[0], 0.5, places=5)

    def test_softmax_activation(self):
        """测试Softmax激活函数"""
        x = tf.constant([[1.0, 2.0, 3.0]])
        output = tf.nn.softmax(x)
        # Softmax输出应该和为1
        self.assertAlmostEqual(tf.reduce_sum(output).numpy(), 1.0, places=5)


class TestCustomLayers(unittest.TestCase):
    """测试自定义层"""

    def test_custom_layer_creation(self):
        """测试自定义层创建"""

        class CustomDense(tf.keras.layers.Layer):
            def __init__(self, units):
                super(CustomDense, self).__init__()
                self.units = units

            def build(self, input_shape):
                self.w = self.add_weight(
                    shape=(input_shape[-1], self.units), initializer="random_normal", trainable=True
                )
                self.b = self.add_weight(shape=(self.units,), initializer="zeros", trainable=True)

            def call(self, inputs):
                return tf.matmul(inputs, self.w) + self.b

        layer = CustomDense(5)
        input_data = tf.random.normal((2, 10))
        output = layer(input_data)
        self.assertEqual(output.shape, (2, 5))


if __name__ == "__main__":
    unittest.main()
