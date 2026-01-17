"""
测试 06_Utils 模块的功能
"""

import unittest
import tensorflow as tf
import numpy as np
import sys
import os
import tempfile
import matplotlib

matplotlib.use("Agg")  # 使用非交互式后端
import matplotlib.pyplot as plt

# 添加项目根目录到路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# 使用importlib导入06_Utils模块（Python模块名不能以数字开头）
import importlib.util

spec = importlib.util.spec_from_file_location(
    "learning_tools", os.path.join(os.path.dirname(__file__), "..", "06_Utils", "learning_tools.py")
)
learning_tools = importlib.util.module_from_spec(spec)
spec.loader.exec_module(learning_tools)

count_parameters = learning_tools.count_parameters


class TestModelUtilities(unittest.TestCase):
    """测试模型工具函数"""

    def setUp(self):
        """测试前的准备工作"""
        tf.random.set_seed(42)
        np.random.seed(42)

    def test_count_parameters(self):
        """测试参数计数"""
        model = tf.keras.Sequential(
            [tf.keras.layers.Dense(10, input_shape=(5,)), tf.keras.layers.Dense(1)]
        )

        # 构建模型
        model.build((None, 5))

        params_info = count_parameters(model)

        self.assertIn("trainable_parameters", params_info)
        self.assertIn("non_trainable_parameters", params_info)
        self.assertIn("total_parameters", params_info)
        self.assertGreater(params_info["total_parameters"], 0)


class TestVisualizationTools(unittest.TestCase):
    """测试可视化工具"""

    def setUp(self):
        """测试前的准备工作"""
        tf.random.set_seed(42)
        np.random.seed(42)
        plt.close("all")

    def tearDown(self):
        """测试后的清理工作"""
        plt.close("all")

    def test_plot_training_history_creation(self):
        """测试训练历史绘图（不实际显示）"""
        # 创建模拟训练历史
        history_dict = {
            "loss": [0.5, 0.4, 0.3, 0.2],
            "accuracy": [0.7, 0.8, 0.85, 0.9],
            "val_loss": [0.6, 0.5, 0.4, 0.3],
            "val_accuracy": [0.65, 0.75, 0.8, 0.85],
        }

        # 创建模拟History对象
        class MockHistory:
            def __init__(self, history_dict):
                self.history = history_dict

        mock_history = MockHistory(history_dict)

        # 测试绘图功能（保存到临时文件）
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = os.path.join(tmpdir, "test_plot.png")

            # 导入并测试
            plot_training_history = learning_tools.plot_training_history
            plot_training_history(mock_history, save_path=save_path)

            # 检查文件是否创建
            self.assertTrue(os.path.exists(save_path))


class TestDataProcessingUtils(unittest.TestCase):
    """测试数据处理工具"""

    def test_normalize_data(self):
        """测试数据归一化"""
        data = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])

        # 标准化
        mean = np.mean(data, axis=0)
        std = np.std(data, axis=0)
        normalized = (data - mean) / std

        # 验证归一化后的均值接近0
        normalized_mean = np.mean(normalized, axis=0)
        np.testing.assert_array_almost_equal(normalized_mean, [0.0, 0.0], decimal=5)

    def test_split_data(self):
        """测试数据分割"""
        X = np.random.randn(100, 5)
        y = np.random.randint(0, 2, 100)

        # 分割数据
        split_idx = int(0.8 * len(X))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]

        self.assertEqual(len(X_train), 80)
        self.assertEqual(len(X_test), 20)
        self.assertEqual(len(y_train), 80)
        self.assertEqual(len(y_test), 20)


class TestMetricsCalculation(unittest.TestCase):
    """测试指标计算"""

    def test_accuracy_calculation(self):
        """测试准确率计算"""
        y_true = np.array([0, 1, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 0, 1])

        accuracy = np.mean(y_true == y_pred)
        self.assertEqual(accuracy, 0.8)

    def test_confusion_matrix_calculation(self):
        """测试混淆矩阵计算"""
        from sklearn.metrics import confusion_matrix

        y_true = np.array([0, 1, 1, 0, 1, 0])
        y_pred = np.array([0, 1, 0, 0, 1, 1])

        cm = confusion_matrix(y_true, y_pred)

        self.assertEqual(cm.shape, (2, 2))
        # 对角线元素应该是正确预测的数量
        self.assertGreaterEqual(cm[0, 0], 0)
        self.assertGreaterEqual(cm[1, 1], 0)


if __name__ == "__main__":
    unittest.main()
