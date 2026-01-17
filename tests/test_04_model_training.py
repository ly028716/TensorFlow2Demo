"""
测试 04_Model_Training 模块的功能
"""

import unittest
import tensorflow as tf
import numpy as np
import sys
import os
import tempfile

# 添加项目根目录到路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


class TestModelCompilation(unittest.TestCase):
    """测试模型编译"""

    def setUp(self):
        """测试前的准备工作"""
        tf.random.set_seed(42)
        np.random.seed(42)

    def test_basic_compilation(self):
        """测试基本编译"""
        model = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(10, activation="relu", input_shape=(5,)),
                tf.keras.layers.Dense(1, activation="sigmoid"),
            ]
        )

        model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

        # 测试模型已编译
        self.assertIsNotNone(model.optimizer)
        self.assertIsNotNone(model.loss)

    def test_custom_optimizer(self):
        """测试自定义优化器"""
        model = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(10, activation="relu", input_shape=(5,)),
                tf.keras.layers.Dense(1, activation="sigmoid"),
            ]
        )

        custom_optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        model.compile(optimizer=custom_optimizer, loss="binary_crossentropy", metrics=["accuracy"])

        self.assertAlmostEqual(model.optimizer.learning_rate.numpy(), 0.001, places=5)


class TestModelTraining(unittest.TestCase):
    """测试模型训练"""

    def setUp(self):
        """测试前的准备工作"""
        tf.random.set_seed(42)
        np.random.seed(42)

    def test_basic_training(self):
        """测试基本训练"""
        # 创建简单数据
        X = np.random.randn(100, 5).astype(np.float32)
        y = np.random.randint(0, 2, (100, 1)).astype(np.float32)

        # 创建模型
        model = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(10, activation="relu", input_shape=(5,)),
                tf.keras.layers.Dense(1, activation="sigmoid"),
            ]
        )

        model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

        # 训练模型
        history = model.fit(X, y, epochs=2, verbose=0)

        # 测试训练历史
        self.assertIn("loss", history.history)
        self.assertIn("accuracy", history.history)
        self.assertEqual(len(history.history["loss"]), 2)

    def test_training_with_validation(self):
        """测试带验证集的训练"""
        X = np.random.randn(100, 5).astype(np.float32)
        y = np.random.randint(0, 2, (100, 1)).astype(np.float32)

        model = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(10, activation="relu", input_shape=(5,)),
                tf.keras.layers.Dense(1, activation="sigmoid"),
            ]
        )

        model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

        history = model.fit(X, y, epochs=2, validation_split=0.2, verbose=0)

        # 测试验证集指标
        self.assertIn("val_loss", history.history)
        self.assertIn("val_accuracy", history.history)


class TestCallbacks(unittest.TestCase):
    """测试回调函数"""

    def setUp(self):
        """测试前的准备工作"""
        tf.random.set_seed(42)
        np.random.seed(42)

    def test_early_stopping(self):
        """测试早停回调"""
        X = np.random.randn(100, 5).astype(np.float32)
        y = np.random.randint(0, 2, (100, 1)).astype(np.float32)

        model = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(10, activation="relu", input_shape=(5,)),
                tf.keras.layers.Dense(1, activation="sigmoid"),
            ]
        )

        model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor="loss", patience=2, restore_best_weights=True
        )

        history = model.fit(X, y, epochs=10, callbacks=[early_stopping], verbose=0)

        # 早停应该在10个epoch之前停止
        self.assertLessEqual(len(history.history["loss"]), 10)

    def test_model_checkpoint(self):
        """测试模型检查点回调"""
        X = np.random.randn(100, 5).astype(np.float32)
        y = np.random.randint(0, 2, (100, 1)).astype(np.float32)

        model = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(10, activation="relu", input_shape=(5,)),
                tf.keras.layers.Dense(1, activation="sigmoid"),
            ]
        )

        model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = os.path.join(tmpdir, "model.h5")
            checkpoint = tf.keras.callbacks.ModelCheckpoint(
                checkpoint_path, save_best_only=True, monitor="loss"
            )

            model.fit(X, y, epochs=2, callbacks=[checkpoint], verbose=0)

            # 检查文件是否创建
            self.assertTrue(os.path.exists(checkpoint_path))


class TestModelEvaluation(unittest.TestCase):
    """测试模型评估"""

    def setUp(self):
        """测试前的准备工作"""
        tf.random.set_seed(42)
        np.random.seed(42)

    def test_model_evaluate(self):
        """测试模型评估"""
        X_train = np.random.randn(100, 5).astype(np.float32)
        y_train = np.random.randint(0, 2, (100, 1)).astype(np.float32)
        X_test = np.random.randn(20, 5).astype(np.float32)
        y_test = np.random.randint(0, 2, (20, 1)).astype(np.float32)

        model = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(10, activation="relu", input_shape=(5,)),
                tf.keras.layers.Dense(1, activation="sigmoid"),
            ]
        )

        model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

        model.fit(X_train, y_train, epochs=2, verbose=0)

        # 评估模型
        loss, accuracy = model.evaluate(X_test, y_test, verbose=0)

        self.assertIsInstance(loss, float)
        self.assertIsInstance(accuracy, float)
        self.assertGreaterEqual(accuracy, 0.0)
        self.assertLessEqual(accuracy, 1.0)

    def test_model_predict(self):
        """测试模型预测"""
        X_train = np.random.randn(100, 5).astype(np.float32)
        y_train = np.random.randint(0, 2, (100, 1)).astype(np.float32)
        X_test = np.random.randn(10, 5).astype(np.float32)

        model = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(10, activation="relu", input_shape=(5,)),
                tf.keras.layers.Dense(1, activation="sigmoid"),
            ]
        )

        model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

        model.fit(X_train, y_train, epochs=2, verbose=0)

        # 预测
        predictions = model.predict(X_test, verbose=0)

        self.assertEqual(predictions.shape, (10, 1))
        # 预测值应该在[0, 1]范围内（sigmoid输出）
        self.assertTrue(np.all(predictions >= 0.0))
        self.assertTrue(np.all(predictions <= 1.0))


class TestModelSaveLoad(unittest.TestCase):
    """测试模型保存和加载"""

    def setUp(self):
        """测试前的准备工作"""
        tf.random.set_seed(42)
        np.random.seed(42)

    def test_save_and_load_model(self):
        """测试保存和加载完整模型"""
        X = np.random.randn(100, 5).astype(np.float32)
        y = np.random.randint(0, 2, (100, 1)).astype(np.float32)

        # 创建和训练模型
        model = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(10, activation="relu", input_shape=(5,)),
                tf.keras.layers.Dense(1, activation="sigmoid"),
            ]
        )

        model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

        model.fit(X, y, epochs=2, verbose=0)

        # 保存模型
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = os.path.join(tmpdir, "test_model.h5")
            model.save(model_path)

            # 加载模型
            loaded_model = tf.keras.models.load_model(model_path)

            # 测试加载的模型
            original_pred = model.predict(X[:5], verbose=0)
            loaded_pred = loaded_model.predict(X[:5], verbose=0)

            np.testing.assert_array_almost_equal(original_pred, loaded_pred, decimal=5)


if __name__ == "__main__":
    unittest.main()
