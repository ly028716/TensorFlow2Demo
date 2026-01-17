"""
集成测试 - 测试完整的工作流程
"""

import unittest
import tensorflow as tf
import numpy as np
import sys
import os
import tempfile

# 添加项目根目录到路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


class TestCompleteWorkflow(unittest.TestCase):
    """测试完整的机器学习工作流程"""

    def setUp(self):
        """测试前的准备工作"""
        tf.random.set_seed(42)
        np.random.seed(42)

    def test_end_to_end_classification(self):
        """测试端到端的分类任务"""
        # 1. 生成数据
        X = np.random.randn(200, 10).astype(np.float32)
        y = (np.sum(X, axis=1) > 0).astype(np.int32)

        # 2. 数据预处理
        # 标准化
        mean = np.mean(X, axis=0)
        std = np.std(X, axis=0)
        X_normalized = (X - mean) / (std + 1e-8)

        # 分割数据集
        split_idx = int(0.8 * len(X))
        X_train, X_test = X_normalized[:split_idx], X_normalized[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]

        # 3. 创建数据集
        train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
        train_dataset = train_dataset.shuffle(100).batch(32)

        test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test))
        test_dataset = test_dataset.batch(32)

        # 4. 构建模型
        model = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(32, activation="relu", input_shape=(10,)),
                tf.keras.layers.Dropout(0.2),
                tf.keras.layers.Dense(16, activation="relu"),
                tf.keras.layers.Dense(1, activation="sigmoid"),
            ]
        )

        # 5. 编译模型
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss="binary_crossentropy",
            metrics=["accuracy"],
        )

        # 6. 训练模型
        history = model.fit(train_dataset, epochs=5, validation_data=test_dataset, verbose=0)

        # 7. 评估模型
        loss, accuracy = model.evaluate(test_dataset, verbose=0)

        # 8. 预测
        predictions = model.predict(X_test, verbose=0)
        predicted_classes = (predictions > 0.5).astype(int).flatten()

        # 9. 保存和加载模型
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = os.path.join(tmpdir, "test_model.h5")
            model.save(model_path)
            loaded_model = tf.keras.models.load_model(model_path)

            # 验证加载的模型
            loaded_predictions = loaded_model.predict(X_test, verbose=0)
            np.testing.assert_array_almost_equal(predictions, loaded_predictions, decimal=5)

        # 断言
        self.assertIn("loss", history.history)
        self.assertIn("accuracy", history.history)
        self.assertIn("val_loss", history.history)
        self.assertIn("val_accuracy", history.history)
        self.assertIsInstance(loss, float)
        self.assertIsInstance(accuracy, float)
        self.assertEqual(len(predicted_classes), len(y_test))

    def test_cnn_image_classification_workflow(self):
        """测试CNN图像分类工作流程"""
        # 1. 生成模拟图像数据
        X = np.random.randn(100, 28, 28, 1).astype(np.float32)
        y = np.random.randint(0, 10, 100)

        # 2. 归一化
        X = (X - np.mean(X)) / (np.std(X) + 1e-8)

        # 3. One-hot编码
        y_onehot = tf.keras.utils.to_categorical(y, 10)

        # 4. 分割数据
        split_idx = int(0.8 * len(X))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y_onehot[:split_idx], y_onehot[split_idx:]

        # 5. 构建CNN模型
        model = tf.keras.Sequential(
            [
                tf.keras.layers.Conv2D(16, (3, 3), activation="relu", input_shape=(28, 28, 1)),
                tf.keras.layers.MaxPooling2D((2, 2)),
                tf.keras.layers.Conv2D(32, (3, 3), activation="relu"),
                tf.keras.layers.MaxPooling2D((2, 2)),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(64, activation="relu"),
                tf.keras.layers.Dropout(0.5),
                tf.keras.layers.Dense(10, activation="softmax"),
            ]
        )

        # 6. 编译
        model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

        # 7. 训练
        history = model.fit(
            X_train, y_train, epochs=3, batch_size=16, validation_split=0.2, verbose=0
        )

        # 8. 评估
        loss, accuracy = model.evaluate(X_test, y_test, verbose=0)

        # 9. 预测
        predictions = model.predict(X_test, verbose=0)
        predicted_classes = np.argmax(predictions, axis=1)

        # 断言
        self.assertEqual(predictions.shape, (len(X_test), 10))
        self.assertEqual(len(predicted_classes), len(X_test))
        self.assertGreaterEqual(accuracy, 0.0)
        self.assertLessEqual(accuracy, 1.0)


class TestDataPipeline(unittest.TestCase):
    """测试数据处理流水线"""

    def test_complete_data_pipeline(self):
        """测试完整的数据处理流水线"""
        # 1. 创建原始数据
        raw_data = np.random.randn(1000, 5).astype(np.float32)
        raw_labels = np.random.randint(0, 3, 1000)

        # 2. 创建Dataset
        dataset = tf.data.Dataset.from_tensor_slices((raw_data, raw_labels))

        # 3. 应用转换
        def preprocess(x, y):
            # 标准化
            x = (x - tf.reduce_mean(x)) / (tf.math.reduce_std(x) + 1e-8)
            # One-hot编码
            y = tf.one_hot(y, depth=3)
            return x, y

        dataset = dataset.map(preprocess)

        # 4. 混洗和批处理
        dataset = dataset.shuffle(buffer_size=100)
        dataset = dataset.batch(32)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)

        # 5. 验证数据集
        for batch_x, batch_y in dataset.take(1):
            self.assertEqual(batch_x.shape[0], 32)
            self.assertEqual(batch_x.shape[1], 5)
            self.assertEqual(batch_y.shape, (32, 3))


class TestCustomTrainingLoop(unittest.TestCase):
    """测试自定义训练循环"""

    def test_custom_training_loop(self):
        """测试自定义训练循环"""
        # 1. 准备数据
        X = np.random.randn(100, 5).astype(np.float32)
        y = np.random.randint(0, 2, (100, 1)).astype(np.float32)

        dataset = tf.data.Dataset.from_tensor_slices((X, y))
        dataset = dataset.batch(16)

        # 2. 创建模型
        model = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(10, activation="relu", input_shape=(5,)),
                tf.keras.layers.Dense(1, activation="sigmoid"),
            ]
        )

        # 3. 定义损失函数和优化器
        loss_fn = tf.keras.losses.BinaryCrossentropy()
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)

        # 4. 定义训练步骤
        @tf.function
        def train_step(x, y):
            with tf.GradientTape() as tape:
                predictions = model(x, training=True)
                loss = loss_fn(y, predictions)

            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))

            return loss

        # 5. 训练循环
        losses = []
        for epoch in range(3):
            epoch_losses = []
            for batch_x, batch_y in dataset:
                loss = train_step(batch_x, batch_y)
                epoch_losses.append(loss.numpy())
            losses.append(np.mean(epoch_losses))

        # 6. 验证
        self.assertEqual(len(losses), 3)
        # 损失应该是有限的数值
        for loss in losses:
            self.assertTrue(np.isfinite(loss))


if __name__ == "__main__":
    unittest.main()
