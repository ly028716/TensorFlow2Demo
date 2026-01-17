"""
TensorFlow 2.0 图像分类实用案例

本模块展示如何使用TensorFlow 2.0构建和训练图像分类模型
包括CNN架构、数据增强、迁移学习等技术
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator


# =============================
# 1. 基础CNN图像分类
# =============================


def build_basic_cnn(input_shape=(32, 32, 3), num_classes=10):
    """
    构建基础CNN模型
    """
    model = tf.keras.Sequential(
        [
            # 第一卷积块
            tf.keras.layers.Conv2D(
                32, (3, 3), activation="relu", padding="same", input_shape=input_shape
            ),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Dropout(0.25),
            # 第二卷积块
            tf.keras.layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Dropout(0.25),
            # 第三卷积块
            tf.keras.layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Dropout(0.25),
            # 全连接层
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(512, activation="relu"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(num_classes, activation="softmax"),
        ]
    )

    return model


def basic_cnn_classification():
    """
    基础CNN图像分类示例 (使用CIFAR-10数据集)
    """
    print("=" * 50)
    print("基础CNN图像分类 (CIFAR-10)")
    print("=" * 50)

    # 加载CIFAR-10数据集
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    print(f"训练数据形状: {x_train.shape}")
    print(f"测试数据形状: {x_test.shape}")

    # 数据预处理
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    # 将标签转换为one-hot编码
    y_train = tf.keras.utils.to_categorical(y_train, 10)
    y_test = tf.keras.utils.to_categorical(y_test, 10)

    # 分割验证集
    x_val = x_train[:5000]
    y_val = y_train[:5000]
    x_train = x_train[5000:]
    y_train = y_train[5000:]

    print(f"训练集: {x_train.shape}")
    print(f"验证集: {x_val.shape}")
    print(f"测试集: {x_test.shape}")

    # 构建模型
    model = build_basic_cnn(input_shape=(32, 32, 3), num_classes=10)
    model.summary()

    # 编译模型
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    # 数据增强
    datagen = ImageDataGenerator(
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        zoom_range=0.1,
    )

    # 训练回调
    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5, min_lr=0.00001),
    ]

    # 训练模型
    print("\n开始训练...")
    history = model.fit(
        datagen.flow(x_train, y_train, batch_size=64),
        epochs=50,
        validation_data=(x_val, y_val),
        callbacks=callbacks,
        verbose=1,
    )

    # 评估模型
    print("\n评估模型...")
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"测试准确率: {test_acc:.4f}")

    # 可视化结果
    plot_training_history(history, "basic_cnn_history.png")

    return model, history


# =============================
# 2. 迁移学习图像分类
# =============================


def transfer_learning_classification():
    """
    迁移学习图像分类示例 (使用预训练MobileNetV2)
    """
    print("\n" + "=" * 50)
    print("迁移学习图像分类 (MobileNetV2)")
    print("=" * 50)

    # 加载CIFAR-10数据集
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

    # 数据预处理 (MobileNetV2需要特定预处理)
    def preprocess_mobilenet(images):
        # 调整大小到96x96 (减少计算量)
        images_resized = tf.image.resize(images, (96, 96))
        # MobileNetV2预处理
        return tf.keras.applications.mobilenet_v2.preprocess_input(images_resized)

    x_train_processed = preprocess_mobilenet(x_train)
    x_test_processed = preprocess_mobilenet(x_test)

    # 将标签转换为one-hot编码
    y_train = tf.keras.utils.to_categorical(y_train, 10)
    y_test = tf.keras.utils.to_categorical(y_test, 10)

    # 分割验证集
    x_val = x_train_processed[:5000]
    y_val = y_train[:5000]
    x_train = x_train_processed[5000:]
    y_train = y_train[5000:]

    # 创建数据增强
    train_datagen = ImageDataGenerator(
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        zoom_range=0.1,
    )

    # 加载预训练的MobileNetV2模型
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(96, 96, 3), include_top=False, weights="imagenet"  # 不包括顶部分类层
    )

    # 冻结预训练权重
    base_model.trainable = False

    # 创建自定义分类头
    inputs = tf.keras.Input(shape=(96, 96, 3))
    x = base_model(inputs, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.1)(x)
    outputs = tf.keras.layers.Dense(10, activation="softmax")(x)

    model = tf.keras.Model(inputs, outputs)

    # 第一次编译 (只训练分类头)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    print("第一阶段训练 (只训练分类头)...")
    model.summary()

    # 训练分类头
    history_head = model.fit(
        train_datagen.flow(x_train, y_train, batch_size=32),
        epochs=10,
        validation_data=(x_val, y_val),
        verbose=1,
    )

    # 解冻部分顶层进行微调
    print("\n第二阶段训练 (微调)...")
    base_model.trainable = True
    # 只解冻最后几层
    fine_tune_at = len(base_model.layers) - 20
    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False

    # 重新编译 (使用更小的学习率)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    # 微调模型
    history_fine_tune = model.fit(
        train_datagen.flow(x_train, y_train, batch_size=32),
        epochs=20,
        validation_data=(x_val, y_val),
        callbacks=[tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)],
        verbose=1,
    )

    # 评估模型
    print("\n评估模型...")
    test_loss, test_acc = model.evaluate(x_test_processed, y_test, verbose=0)
    print(f"测试准确率: {test_acc:.4f}")

    # 可视化结果
    plot_training_history(history_head, "transfer_learning_head.png")
    plot_training_history(history_fine_tune, "transfer_learning_fine_tune.png")

    return model, history_head, history_fine_tune


# =============================
# 3. 自定义CNN架构
# =============================


class ResidualBlock(tf.keras.layers.Layer):
    """
    残差块实现
    """

    def __init__(self, filters, stride=1, **kwargs):
        super(ResidualBlock, self).__init__(**kwargs)
        self.filters = filters
        self.stride = stride

        # 主路径
        self.conv1 = tf.keras.layers.Conv2D(
            filters, kernel_size=3, strides=stride, padding="same", use_bias=False
        )
        self.bn1 = tf.keras.layers.BatchNormalization()

        self.conv2 = tf.keras.layers.Conv2D(
            filters, kernel_size=3, strides=1, padding="same", use_bias=False
        )
        self.bn2 = tf.keras.layers.BatchNormalization()

        # 捷径路径
        if stride != 1:
            self.shortcut = tf.keras.Sequential(
                [
                    tf.keras.layers.Conv2D(
                        filters, kernel_size=1, strides=stride, padding="same", use_bias=False
                    ),
                    tf.keras.layers.BatchNormalization(),
                ]
            )
        else:
            self.shortcut = lambda x: x

    def call(self, inputs, training=None):
        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = tf.nn.relu(x)

        x = self.conv2(x)
        x = self.bn2(x, training=training)

        # 添加残差连接
        x += self.shortcut(inputs)
        return tf.nn.relu(x)


def build_resnet(input_shape=(32, 32, 3), num_classes=10):
    """
    构建简化版ResNet
    """
    inputs = tf.keras.Input(shape=input_shape)

    # 初始层
    x = tf.keras.layers.Conv2D(64, kernel_size=3, strides=1, padding="same")(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.nn.relu(x)

    # 残差块
    x = ResidualBlock(64)(x)
    x = ResidualBlock(64)(x)

    x = ResidualBlock(128, stride=2)(x)
    x = ResidualBlock(128)(x)

    x = ResidualBlock(256, stride=2)(x)
    x = ResidualBlock(256)(x)

    # 分类层
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)

    return tf.keras.Model(inputs, outputs)


def custom_resnet_classification():
    """
    自定义ResNet图像分类示例
    """
    print("\n" + "=" * 50)
    print("自定义ResNet图像分类")
    print("=" * 50)

    # 加载CIFAR-10数据集
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

    # 数据预处理
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    # 数据标准化
    mean = np.mean(x_train, axis=0)
    std = np.std(x_train, axis=0)
    x_train = (x_train - mean) / (std + 1e-7)
    x_test = (x_test - mean) / (std + 1e-7)

    # 标签处理
    y_train = tf.keras.utils.to_categorical(y_train, 10)
    y_test = tf.keras.utils.to_categorical(y_test, 10)

    # 分割验证集
    x_val = x_train[:5000]
    y_val = y_train[:5000]
    x_train = x_train[5000:]
    y_train = y_train[5000:]

    # 构建模型
    model = build_resnet(input_shape=(32, 32, 3), num_classes=10)
    model.summary()

    # 编译模型
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    # 数据增强
    train_datagen = ImageDataGenerator(
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        zoom_range=0.1,
    )

    # 训练回调
    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=15, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=7, min_lr=0.00001),
    ]

    # 训练模型
    print("\n开始训练...")
    history = model.fit(
        train_datagen.flow(x_train, y_train, batch_size=64),
        epochs=50,
        validation_data=(x_val, y_val),
        callbacks=callbacks,
        verbose=1,
    )

    # 评估模型
    print("\n评估模型...")
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"测试准确率: {test_acc:.4f}")

    # 可视化结果
    plot_training_history(history, "custom_resnet_history.png")

    return model, history


# =============================
# 4. 可视化工具
# =============================


def plot_training_history(history, filename):
    """
    绘制训练历史
    """
    plt.figure(figsize=(12, 5))

    # 绘制损失曲线
    plt.subplot(1, 2, 1)
    plt.plot(history.history["loss"], label="训练损失")
    plt.plot(history.history["val_loss"], label="验证损失")
    plt.title("模型损失")
    plt.xlabel("Epoch")
    plt.ylabel("损失")
    plt.legend()
    plt.grid(True)

    # 绘制准确率曲线
    plt.subplot(1, 2, 2)
    plt.plot(history.history["accuracy"], label="训练准确率")
    plt.plot(history.history["val_accuracy"], label="验证准确率")
    plt.title("模型准确率")
    plt.xlabel("Epoch")
    plt.ylabel("准确率")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(filename)
    print(f"训练历史图表已保存为 '{filename}'")
    plt.close()


def visualize_predictions(model, x_test, y_test, class_names, num_samples=10):
    """
    可视化预测结果
    """
    # 获取预测结果
    predictions = model.predict(x_test[:num_samples])
    predicted_classes = np.argmax(predictions, axis=1)
    true_classes = np.argmax(y_test[:num_samples], axis=1)

    # 可视化
    plt.figure(figsize=(15, 5))
    for i in range(num_samples):
        plt.subplot(2, 5, i + 1)
        plt.imshow(x_test[i])
        plt.xticks([])
        plt.yticks([])

        # 设置标题颜色 (正确为绿色，错误为红色)
        color = "green" if predicted_classes[i] == true_classes[i] else "red"
        title = f"True: {class_names[true_classes[i]]}\nPred: {class_names[predicted_classes[i]]}"
        plt.title(title, color=color)

    plt.tight_layout()
    plt.savefig("prediction_visualization.png")
    print("预测可视化图表已保存为 'prediction_visualization.png'")
    plt.close()


# =============================
# 5. 实用工具函数
# =============================


def create_class_labels():
    """
    创建CIFAR-10类别标签
    """
    return [
        "airplane",
        "automobile",
        "bird",
        "cat",
        "deer",
        "dog",
        "frog",
        "horse",
        "ship",
        "truck",
    ]


def evaluate_model_performance(model, x_test, y_test):
    """
    全面评估模型性能
    """
    # 预测
    predictions = model.predict(x_test)
    predicted_classes = np.argmax(predictions, axis=1)
    true_classes = np.argmax(y_test, axis=1)

    # 计算混淆矩阵
    from sklearn.metrics import confusion_matrix, classification_report

    cm = confusion_matrix(true_classes, predicted_classes)

    # 打印分类报告
    class_names = create_class_labels()
    print("\n分类报告:")
    print(classification_report(true_classes, predicted_classes, target_names=class_names))

    # 可视化混淆矩阵
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title("混淆矩阵")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    # 添加数值标签
    thresh = cm.max() / 2.0
    for i, j in np.ndindex(cm.shape):
        plt.text(
            j,
            i,
            format(cm[i, j], "d"),
            ha="center",
            va="center",
            color="white" if cm[i, j] > thresh else "black",
        )

    plt.ylabel("真实类别")
    plt.xlabel("预测类别")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png")
    print("混淆矩阵已保存为 'confusion_matrix.png'")
    plt.close()

    return cm


# =============================
# 主函数
# =============================


def main():
    """
    主函数，执行所有图像分类示例
    """
    print("TensorFlow 2.0 图像分类实用案例")
    print("=" * 60)

    try:
        # 1. 基础CNN分类
        print("\n1. 基础CNN分类")
        basic_model, basic_history = basic_cnn_classification()

        # 2. 迁移学习分类
        print("\n2. 迁移学习分类")
        transfer_model, head_history, fine_tune_history = transfer_learning_classification()

        # 3. 自定义ResNet分类
        print("\n3. 自定义ResNet分类")
        resnet_model, resnet_history = custom_resnet_classification()

        # 4. 模型比较
        print("\n4. 模型性能比较:")
        models = [
            ("基础CNN", basic_model, basic_history),
            ("迁移学习", transfer_model, fine_tune_history),
            ("ResNet", resnet_model, resnet_history),
        ]

        # 加载测试数据
        (_, _), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
        x_test = x_test.astype("float32") / 255.0
        y_test_cat = tf.keras.utils.to_categorical(y_test, 10)

        # 为迁移学习模型准备测试数据
        def preprocess_mobilenet(images):
            images_resized = tf.image.resize(images, (96, 96))
            return tf.keras.applications.mobilenet_v2.preprocess_input(images_resized)

        for name, model, history in models:
            if name == "迁移学习":
                x_test_processed = preprocess_mobilenet(x_test)
                test_loss, test_acc = model.evaluate(x_test_processed, y_test_cat, verbose=0)
            else:
                test_loss, test_acc = model.evaluate(x_test, y_test_cat, verbose=0)
            print(f"{name}: 测试准确率 = {test_acc:.4f}")

    except Exception as e:
        print(f"执行过程中出现错误: {e}")
        print("这可能是由于缺少某些依赖或环境配置问题")

    print("\n" + "=" * 60)
    print("图像分类案例学习完成！")
    print("\n关键要点:")
    print("1. CNN是图像分类的基础架构")
    print("2. 数据增强可以提高模型泛化能力")
    print("3. 迁移学习在小数据集上特别有效")
    print("4. 残差连接有助于训练更深的网络")
    print("5. 合理的评估和可视化是必不可少的")


if __name__ == "__main__":
    main()
