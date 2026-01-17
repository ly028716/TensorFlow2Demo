"""
TensorFlow 2.0 学习工具

本模块包含辅助学习的工具函数
包括可视化、评估、数据处理等实用功能
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import time
import os


# =============================
# 1. 可视化工具
# =============================


def plot_training_history(history, metrics=["loss", "accuracy"], save_path=None):
    """
    绘制训练历史

    Args:
        history: 训练历史对象
        metrics: 要绘制的指标列表
        save_path: 保存路径（可选）
    """
    num_metrics = len(metrics)
    fig, axes = plt.subplots(1, num_metrics, figsize=(5 * num_metrics, 4))

    if num_metrics == 1:
        axes = [axes]

    for i, metric in enumerate(metrics):
        axes[i].plot(history.history[metric], label="训练")
        val_metric = f"val_{metric}"
        if val_metric in history.history:
            axes[i].plot(history.history[val_metric], label="验证")

        axes[i].set_title(f"{metric}曲线")
        axes[i].set_xlabel("Epoch")
        axes[i].set_ylabel(metric)
        axes[i].legend()
        axes[i].grid(True)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"训练历史图表已保存到: {save_path}")

    plt.show()


def plot_confusion_matrix(y_true, y_pred, class_names=None, save_path=None):
    """
    绘制混淆矩阵

    Args:
        y_true: 真实标签
        y_pred: 预测标签
        class_names: 类别名称列表
        save_path: 保存路径（可选）
    """
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names
    )

    plt.title("混淆矩阵")
    plt.xlabel("预测类别")
    plt.ylabel("真实类别")

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"混淆矩阵已保存到: {save_path}")

    plt.show()


def plot_model_architecture(model, save_path=None):
    """
    绘制模型架构

    Args:
        model: TensorFlow模型
        save_path: 保存路径（可选）
    """
    if save_path:
        tf.keras.utils.plot_model(
            model, to_file=save_path, show_shapes=True, show_layer_names=True, rankdir="TB", dpi=300
        )
        print(f"模型架构图已保存到: {save_path}")
    else:
        tf.keras.utils.plot_model(model, show_shapes=True, show_layer_names=True, rankdir="TB")


def visualize_predictions(images, predictions, true_labels=None, class_names=None, save_path=None):
    """
    可视化预测结果

    Args:
        images: 图像数据
        predictions: 预测结果
        true_labels: 真实标签（可选）
        class_names: 类别名称（可选）
        save_path: 保存路径（可选）
    """
    num_images = min(len(images), 16)  # 最多显示16张图
    rows = int(np.ceil(num_images / 4))

    plt.figure(figsize=(15, 4 * rows))

    for i in range(num_images):
        plt.subplot(rows, 4, i + 1)

        # 显示图像
        if len(images[i].shape) == 3:  # 彩色图像
            plt.imshow(images[i])
        else:  # 灰度图像
            plt.imshow(images[i], cmap="gray")

        plt.axis("off")

        # 构建标题
        if true_labels is not None:
            true_label = true_labels[i]
            pred_label = np.argmax(predictions[i])
            correct = true_label == pred_label
            color = "green" if correct else "red"

            if class_names:
                title = f"真实: {class_names[true_label]}\n预测: {class_names[pred_label]}"
            else:
                title = f"真实: {true_label}\n预测: {pred_label}"

            plt.title(title, color=color)
        else:
            pred_label = np.argmax(predictions[i])
            conf = np.max(predictions[i])
            if class_names:
                title = f"预测: {class_names[pred_label]}\n置信度: {conf:.2f}"
            else:
                title = f"预测: {pred_label}\n置信度: {conf:.2f}"

            plt.title(title)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"预测可视化已保存到: {save_path}")

    plt.show()


# =============================
# 2. 评估工具
# =============================


def comprehensive_evaluation(model, X_test, y_test, class_names=None):
    """
    全面评估模型性能

    Args:
        model: 训练好的模型
        X_test: 测试数据
        y_test: 测试标签
        class_names: 类别名称（可选）

    Returns:
        dict: 包含各种评估指标的字典
    """
    print("=" * 50)
    print("模型全面评估")
    print("=" * 50)

    # 基本评估
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"测试损失: {test_loss:.4f}")
    print(f"测试准确率: {test_accuracy:.4f}")

    # 预测
    predictions = model.predict(X_test)

    # 处理不同类型的预测
    if len(predictions.shape) == 1:  # 回归任务
        pred_labels = predictions
        y_true = y_test
    else:  # 分类任务
        pred_labels = np.argmax(predictions, axis=1)
        if len(y_test.shape) > 1 and y_test.shape[1] > 1:  # one-hot编码
            y_true = np.argmax(y_test, axis=1)
        else:
            y_true = y_test

    # 分类报告
    if len(predictions.shape) > 1:  # 分类任务
        print("\n分类报告:")
        print(classification_report(y_true, pred_labels, target_names=class_names))

        # 绘制混淆矩阵
        plot_confusion_matrix(y_true, pred_labels, class_names)

    # 计算其他指标
    metrics = {
        "test_loss": test_loss,
        "test_accuracy": test_accuracy,
        "predictions": predictions,
        "pred_labels": pred_labels,
        "y_true": y_true,
    }

    return metrics


def compare_models(models, X_test, y_test, model_names=None):
    """
    比较多个模型的性能

    Args:
        models: 模型列表
        X_test: 测试数据
        y_test: 测试标签
        model_names: 模型名称列表（可选）

    Returns:
        DataFrame: 模型比较结果
    """
    import pandas as pd

    if model_names is None:
        model_names = [f"Model_{i+1}" for i in range(len(models))]

    results = []

    for i, model in enumerate(models):
        test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
        results.append({"Model": model_names[i], "Test Loss": test_loss, "Test Accuracy": test_acc})

    df = pd.DataFrame(results)

    # 排序
    df = df.sort_values("Test Accuracy", ascending=False)

    print("\n模型比较:")
    print("=" * 50)
    print(df.to_string(index=False))

    # 可视化比较
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.bar(df["Model"], df["Test Accuracy"])
    plt.title("模型准确率比较")
    plt.ylabel("准确率")
    plt.xticks(rotation=45)

    plt.subplot(1, 2, 2)
    plt.bar(df["Model"], df["Test Loss"])
    plt.title("模型损失比较")
    plt.ylabel("损失")
    plt.xticks(rotation=45)

    plt.tight_layout()
    plt.show()

    return df


# =============================
# 3. 性能监控工具
# =============================


class TrainingMonitor(tf.keras.callbacks.Callback):
    """
    自定义训练监控回调
    """

    def __init__(self, print_freq=10):
        super(TrainingMonitor, self).__init__()
        self.print_freq = print_freq
        self.start_time = None

    def on_train_begin(self, logs=None):
        self.start_time = time.time()
        print("训练开始...")

    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self.print_freq == 0:
            elapsed = time.time() - self.start_time
            print(
                f"Epoch {epoch+1}: loss={logs['loss']:.4f}, "
                f"accuracy={logs['accuracy']:.4f}, "
                f"val_loss={logs.get('val_loss', 'N/A')}, "
                f"val_accuracy={logs.get('val_accuracy', 'N/A')}, "
                f"time={elapsed:.2f}s"
            )

    def on_train_end(self, logs=None):
        total_time = time.time() - self.start_time
        print(f"训练完成！总耗时: {total_time:.2f}s")


class LearningRateScheduler(tf.keras.callbacks.Callback):
    """
    自定义学习率调度器
    """

    def __init__(self, schedule, verbose=1):
        super(LearningRateScheduler, self).__init__()
        self.schedule = schedule
        self.verbose = verbose

    def on_epoch_begin(self, epoch, logs=None):
        lr = self.schedule(epoch)
        tf.keras.backend.set_value(self.model.optimizer.lr, lr)
        if self.verbose:
            print(f"\nEpoch {epoch+1}: 学习率设置为 {lr:.6f}")


# =============================
# 4. 数据处理工具
# =============================


def create_image_data_generator(
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    zoom_range=0.1,
    fill_mode="nearest",
):
    """
    创建图像数据增强生成器

    Args:
        rotation_range: 旋转角度范围
        width_shift_range: 水平平移范围
        height_shift_range: 垂直平移范围
        horizontal_flip: 是否水平翻转
        zoom_range: 缩放范围
        fill_mode: 填充模式

    Returns:
        ImageDataGenerator: 数据增强生成器
    """
    from tensorflow.keras.preprocessing.image import ImageDataGenerator

    return ImageDataGenerator(
        rotation_range=rotation_range,
        width_shift_range=width_shift_range,
        height_shift_range=height_shift_range,
        horizontal_flip=horizontal_flip,
        zoom_range=zoom_range,
        fill_mode=fill_mode,
    )


def visualize_data_augmentation(generator, sample_image, num_samples=9):
    """
    可视化数据增强效果

    Args:
        generator: 数据增强生成器
        sample_image: 样本图像
        num_samples: 显示的样本数量
    """
    # 扩展维度以适应生成器
    sample_image = np.expand_dims(sample_image, axis=0)

    # 生成增强样本
    augmented_images = []
    for i in range(num_samples):
        batch = next(generator.flow(sample_image, batch_size=1))
        augmented_images.append(batch[0])

    # 可视化
    rows = int(np.sqrt(num_samples))
    cols = int(np.ceil(num_samples / rows))

    plt.figure(figsize=(12, 12))
    for i in range(num_samples):
        plt.subplot(rows, cols, i + 1)
        plt.imshow(augmented_images[i])
        plt.axis("off")

    plt.tight_layout()
    plt.show()


def create_tf_data_pipeline(
    data, labels=None, batch_size=32, shuffle=True, prefetch=True, cache=False
):
    """
    创建高效的数据管道

    Args:
        data: 输入数据
        labels: 标签数据（可选）
        batch_size: 批次大小
        shuffle: 是否打乱数据
        prefetch: 是否预取
        cache: 是否缓存

    Returns:
        tf.data.Dataset: 数据管道
    """
    # 创建数据集
    if labels is not None:
        dataset = tf.data.Dataset.from_tensor_slices((data, labels))
    else:
        dataset = tf.data.Dataset.from_tensor_slices(data)

    # 打乱数据
    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(data))

    # 批处理
    dataset = dataset.batch(batch_size)

    # 缓存
    if cache:
        dataset = dataset.cache()

    # 预取
    if prefetch:
        dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset


# =============================
# 5. 模型工具
# =============================


def count_parameters(model):
    """
    计算模型的参数数量

    Args:
        model: TensorFlow模型

    Returns:
        dict: 包含参数数量的字典
    """
    trainable_params = sum([tf.keras.backend.count_params(w) for w in model.trainable_weights])
    non_trainable_params = sum(
        [tf.keras.backend.count_params(w) for w in model.non_trainable_weights]
    )
    total_params = trainable_params + non_trainable_params

    info = {
        "trainable_parameters": trainable_params,
        "non_trainable_parameters": non_trainable_params,
        "total_parameters": total_params,
    }

    print("模型参数信息:")
    print(f"可训练参数: {trainable_params:,}")
    print(f"不可训练参数: {non_trainable_params:,}")
    print(f"总参数: {total_params:,}")

    return info


def save_model_with_info(model, model_path, history=None, save_format="h5"):
    """
    保存模型及相关信息

    Args:
        model: TensorFlow模型
        model_path: 模型保存路径
        history: 训练历史（可选）
        save_format: 保存格式
    """
    # 确保目录存在
    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    # 保存模型
    model.save(model_path, save_format=save_format)
    print(f"模型已保存到: {model_path}")

    # 保存训练历史
    if history:
        history_path = model_path.replace(f".{save_format}", "_history.npy")
        np.save(history_path, history.history)
        print(f"训练历史已保存到: {history_path}")

    # 保存模型信息
    info_path = model_path.replace(f".{save_format}", "_info.txt")
    with open(info_path, "w") as f:
        f.write("模型架构:\n")
        model.summary(print_fn=lambda x: f.write(x + "\n"))

        f.write("\n参数信息:\n")
        params_info = count_parameters(model)
        for key, value in params_info.items():
            f.write(f"{key}: {value:,}\n")

    print(f"模型信息已保存到: {info_path}")


def load_model_with_info(model_path):
    """
    加载模型及相关信息

    Args:
        model_path: 模型路径

    Returns:
        tuple: (模型, 历史记录)
    """
    # 加载模型
    model = tf.keras.models.load_model(model_path)
    print(f"模型已从 {model_path} 加载")

    # 加载训练历史
    history_path = model_path.replace(".h5", "_history.npy")
    history = None
    if os.path.exists(history_path):
        history_dict = np.load(history_path, allow_pickle=True).item()
        print(f"训练历史已加载")

        # 创建简单的历史对象
        class SimpleHistory:
            def __init__(self, history_dict):
                self.history = history_dict

        history = SimpleHistory(history_dict)

    return model, history


# =============================
# 主函数
# =============================


def main():
    """
    演示学习工具的使用
    """
    print("TensorFlow 2.0 学习工具演示")
    print("=" * 50)

    # 1. 参数计数示例
    print("1. 模型参数计数:")
    sample_model = tf.keras.Sequential(
        [
            tf.keras.layers.Dense(64, activation="relu", input_shape=(10,)),
            tf.keras.layers.Dense(32, activation="relu"),
            tf.keras.layers.Dense(1, activation="sigmoid"),
        ]
    )
    count_parameters(sample_model)

    # 2. 数据管道示例
    print("\n2. 数据管道创建:")
    data = np.random.random((100, 10))
    labels = np.random.randint(0, 2, (100,))
    dataset = create_tf_data_pipeline(data, labels, batch_size=16)
    print(f"数据管道创建成功: {dataset}")

    # 3. 图像数据增强示例
    print("\n3. 图像数据增强:")
    generator = create_image_data_generator()
    sample_image = np.random.random((100, 100, 3))
    print(f"图像数据增强器创建成功")

    # 4. 训练监控器示例
    print("\n4. 自定义回调:")
    monitor = TrainingMonitor(print_freq=5)
    print("训练监控器创建成功")

    print("\n" + "=" * 50)
    print("学习工具演示完成！")


if __name__ == "__main__":
    main()
