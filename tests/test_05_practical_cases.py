"""
测试 05_Practical_Cases 模块的功能
"""

import unittest
import tensorflow as tf
import numpy as np
import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from unittest.mock import patch, MagicMock


class TestImageClassification(unittest.TestCase):
    """测试图像分类模块"""

    def setUp(self):
        """测试前的准备工作"""
        tf.random.set_seed(42)
        np.random.seed(42)

    def test_build_basic_cnn(self):
        """测试构建基础CNN模型"""
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "image_classification",
            os.path.join(
                os.path.dirname(__file__), "..", "05_Practical_Cases", "image_classification.py"
            ),
        )
        image_classification = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(image_classification)
        build_basic_cnn = image_classification.build_basic_cnn

        model = build_basic_cnn(input_shape=(32, 32, 3), num_classes=10)

        # 验证模型结构
        self.assertIsInstance(model, tf.keras.Model)
        self.assertEqual(len(model.layers) > 0, True)

        # 验证输入输出形状
        self.assertEqual(model.input_shape, (None, 32, 32, 3))
        self.assertEqual(model.output_shape, (None, 10))

    def test_cnn_model_prediction(self):
        """测试CNN模型预测"""
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "image_classification",
            os.path.join(
                os.path.dirname(__file__), "..", "05_Practical_Cases", "image_classification.py"
            ),
        )
        image_classification = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(image_classification)
        build_basic_cnn = image_classification.build_basic_cnn

        model = build_basic_cnn(input_shape=(32, 32, 3), num_classes=10)
        model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

        # 创建测试数据
        test_images = np.random.random((5, 32, 32, 3)).astype("float32")

        # 测试预测
        predictions = model.predict(test_images, verbose=0)

        # 验证预测结果
        self.assertEqual(predictions.shape, (5, 10))
        # 验证概率和为1
        np.testing.assert_array_almost_equal(np.sum(predictions, axis=1), np.ones(5), decimal=5)

    def test_cnn_model_training(self):
        """测试CNN模型训练"""
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "image_classification",
            os.path.join(
                os.path.dirname(__file__), "..", "05_Practical_Cases", "image_classification.py"
            ),
        )
        image_classification = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(image_classification)
        build_basic_cnn = image_classification.build_basic_cnn

        model = build_basic_cnn(input_shape=(32, 32, 3), num_classes=10)
        model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

        # 创建小型训练数据
        x_train = np.random.random((50, 32, 32, 3)).astype("float32")
        y_train = tf.keras.utils.to_categorical(np.random.randint(0, 10, 50), 10)

        # 训练一个epoch
        history = model.fit(x_train, y_train, epochs=1, batch_size=10, verbose=0)

        # 验证训练历史
        self.assertIn("loss", history.history)
        self.assertIn("accuracy", history.history)
        self.assertEqual(len(history.history["loss"]), 1)


class TestNLPTextClassification(unittest.TestCase):
    """测试文本分类模块"""

    def setUp(self):
        """测试前的准备工作"""
        tf.random.set_seed(42)
        np.random.seed(42)

    @patch("tensorflow.keras.datasets.imdb.load_data")
    @patch("tensorflow.keras.datasets.imdb.get_word_index")
    def test_prepare_imdb_data(self, mock_word_index, mock_load_data):
        """测试IMDB数据准备"""
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "nlp_text_classification",
            os.path.join(
                os.path.dirname(__file__), "..", "05_Practical_Cases", "nlp_text_classification.py"
            ),
        )
        nlp_text_classification = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(nlp_text_classification)
        prepare_imdb_data = nlp_text_classification.prepare_imdb_data

        # 模拟数据
        mock_x_train = [np.array([1, 2, 3, 4, 5]) for _ in range(100)]
        mock_y_train = np.random.randint(0, 2, 100)
        mock_x_test = [np.array([1, 2, 3]) for _ in range(50)]
        mock_y_test = np.random.randint(0, 2, 50)

        mock_load_data.return_value = ((mock_x_train, mock_y_train), (mock_x_test, mock_y_test))
        mock_word_index.return_value = {"the": 1, "movie": 2, "is": 3}

        # 测试数据准备
        (x_train, y_train), (x_test, y_test), word_index = prepare_imdb_data()

        # 验证数据形状
        self.assertEqual(x_train.shape[0], 100)
        self.assertEqual(x_test.shape[0], 50)
        self.assertEqual(x_train.shape[1], 256)  # max_length
        self.assertIsInstance(word_index, dict)

    def test_embedding_model_structure(self):
        """测试词嵌入模型结构"""
        # 创建简单的嵌入模型
        vocab_size = 1000
        embedding_dim = 50
        max_length = 100

        model = tf.keras.Sequential(
            [
                tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
                tf.keras.layers.GlobalAveragePooling1D(),
                tf.keras.layers.Dense(16, activation="relu"),
                tf.keras.layers.Dropout(0.5),
                tf.keras.layers.Dense(1, activation="sigmoid"),
            ]
        )

        # 构建模型以定义input_shape
        model.build(input_shape=(None, max_length))

        # 验证模型结构
        self.assertIsInstance(model, tf.keras.Model)
        self.assertEqual(model.input_shape, (None, max_length))
        self.assertEqual(model.output_shape, (None, 1))

        # 验证嵌入层
        embedding_layer = model.layers[0]
        self.assertIsInstance(embedding_layer, tf.keras.layers.Embedding)
        self.assertEqual(embedding_layer.input_dim, vocab_size)
        self.assertEqual(embedding_layer.output_dim, embedding_dim)

    def test_text_model_prediction(self):
        """测试文本模型预测"""
        vocab_size = 1000
        embedding_dim = 50
        max_length = 100

        model = tf.keras.Sequential(
            [
                tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
                tf.keras.layers.GlobalAveragePooling1D(),
                tf.keras.layers.Dense(16, activation="relu"),
                tf.keras.layers.Dense(1, activation="sigmoid"),
            ]
        )

        model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

        # 创建测试数据
        test_sequences = np.random.randint(0, vocab_size, (10, max_length))

        # 测试预测
        predictions = model.predict(test_sequences, verbose=0)

        # 验证预测结果
        self.assertEqual(predictions.shape, (10, 1))
        # 验证预测值在0-1之间
        self.assertTrue(np.all(predictions >= 0))
        self.assertTrue(np.all(predictions <= 1))

    def test_text_model_training(self):
        """测试文本模型训练"""
        vocab_size = 1000
        embedding_dim = 32
        max_length = 50

        model = tf.keras.Sequential(
            [
                tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
                tf.keras.layers.GlobalAveragePooling1D(),
                tf.keras.layers.Dense(16, activation="relu"),
                tf.keras.layers.Dense(1, activation="sigmoid"),
            ]
        )

        model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

        # 创建小型训练数据
        x_train = np.random.randint(0, vocab_size, (100, max_length))
        y_train = np.random.randint(0, 2, 100)

        # 训练一个epoch
        history = model.fit(x_train, y_train, epochs=1, batch_size=32, verbose=0)

        # 验证训练历史
        self.assertIn("loss", history.history)
        self.assertIn("accuracy", history.history)
        self.assertEqual(len(history.history["loss"]), 1)


class TestDataAugmentation(unittest.TestCase):
    """测试数据增强功能"""

    def test_image_data_generator(self):
        """测试图像数据生成器"""
        from tensorflow.keras.preprocessing.image import ImageDataGenerator

        datagen = ImageDataGenerator(
            rotation_range=20,
            width_shift_range=0.1,
            height_shift_range=0.1,
            horizontal_flip=True,
            zoom_range=0.1,
        )

        # 创建测试图像
        test_image = np.random.random((1, 32, 32, 3)).astype("float32")

        # 测试数据生成
        augmented_images = []
        for batch in datagen.flow(test_image, batch_size=1):
            augmented_images.append(batch)
            if len(augmented_images) >= 5:
                break

        # 验证生成的图像
        self.assertEqual(len(augmented_images), 5)
        for img in augmented_images:
            self.assertEqual(img.shape, (1, 32, 32, 3))


if __name__ == "__main__":
    unittest.main()
