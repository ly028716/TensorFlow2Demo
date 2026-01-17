"""
TensorFlow 2.0 神经网络基础

本模块介绍使用TensorFlow 2.0构建神经网络的基础知识
包括Keras API、层(Layers)、激活函数、损失函数等核心概念
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons, make_classification
from sklearn.model_selection import train_test_split


# =============================
# 1. Keras API 简介
# =============================


def keras_api_introduction():
    """
    介绍TensorFlow 2.0中的Keras API
    """
    print("=" * 50)
    print("Keras API 简介")
    print("=" * 50)

    print("TensorFlow 2.0 集成了Keras作为高级神经网络API")
    print("Keras提供了三种主要的模型构建方式:")
    print("1. Sequential API - 顺序模型，适合简单的层堆叠")
    print("2. Functional API - 函数式API，适合复杂模型结构")
    print("3. Model Subclassing - 模型子类化，完全自定义")

    # 示例数据
    sample_input = tf.random.normal((2, 10))

    # 1. Sequential API
    print("\n1. Sequential API 示例:")
    sequential_model = tf.keras.Sequential(
        [
            tf.keras.layers.Dense(32, activation="relu", input_shape=(10,)),
            tf.keras.layers.Dense(16, activation="relu"),
            tf.keras.layers.Dense(1, activation="sigmoid"),
        ]
    )

    sequential_output = sequential_model(sample_input)
    print(f"Sequential模型输出形状: {sequential_output.shape}")

    # 2. Functional API
    print("\n2. Functional API 示例:")
    inputs = tf.keras.Input(shape=(10,))
    x = tf.keras.layers.Dense(32, activation="relu")(inputs)
    x = tf.keras.layers.Dense(16, activation="relu")(x)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)

    functional_model = tf.keras.Model(inputs=inputs, outputs=outputs)
    functional_output = functional_model(sample_input)
    print(f"Functional模型输出形状: {functional_output.shape}")

    # 3. Model Subclassing
    print("\n3. Model Subclassing 示例:")

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

    subclassed_model = CustomModel()
    subclassed_output = subclassed_model(sample_input)
    print(f"Subclassed模型输出形状: {subclassed_output.shape}")


# =============================
# 2. 神经网络层(Layers)详解
# =============================


def neural_network_layers():
    """
    详细介绍常用的神经网络层
    """
    print("\n" + "=" * 50)
    print("神经网络层详解")
    print("=" * 50)

    # 示例输入
    sample_input = tf.random.normal((2, 10))

    # 1. Dense层 - 全连接层
    print("1. Dense层 (全连接层):")
    dense_layer = tf.keras.layers.Dense(
        units=5,  # 输出维度
        activation="relu",  # 激活函数
        use_bias=True,  # 是否使用偏置
        kernel_regularizer=tf.keras.regularizers.l2(0.01),  # 正则化
    )

    dense_output = dense_layer(sample_input)
    print(f"输入形状: {sample_input.shape}")
    print(f"输出形状: {dense_output.shape}")
    print(f"权重形状: {dense_layer.kernel.shape}")
    print(f"偏置形状: {dense_layer.bias.shape}")

    # 2. Dropout层 - 防止过拟合
    print("\n2. Dropout层:")
    dropout_layer = tf.keras.layers.Dropout(rate=0.5)
    dropout_output = dropout_layer(dense_output, training=True)
    print(f"Dropout前: {dense_output[0, :3].numpy()}")
    print(f"Dropout后: {dropout_output[0, :3].numpy()} (部分元素被置零)")

    # 3. BatchNormalization层 - 批量标准化
    print("\n3. BatchNormalization层:")
    bn_layer = tf.keras.layers.BatchNormalization()
    bn_output = bn_layer(dense_output)
    print(f"BN前均值: {tf.reduce_mean(dense_output).numpy():.4f}")
    print(f"BN后均值: {tf.reduce_mean(bn_output).numpy():.4f}")

    # 4. Conv2D层 - 2D卷积层 (用于图像)
    print("\n4. Conv2D层 (2D卷积):")
    image_input = tf.random.normal((2, 28, 28, 1))  # 批次大小, 高度, 宽度, 通道
    conv_layer = tf.keras.layers.Conv2D(
        filters=32,  # 卷积核数量
        kernel_size=(3, 3),  # 卷积核大小
        strides=(1, 1),  # 步长
        padding="same",  # 填充方式
        activation="relu",
    )

    conv_output = conv_layer(image_input)
    print(f"输入形状: {image_input.shape}")
    print(f"输出形状: {conv_output.shape}")
    print(f"卷积核形状: {conv_layer.kernel.shape}")

    # 5. MaxPooling2D层 - 最大池化
    print("\n5. MaxPooling2D层 (最大池化):")
    pool_layer = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))
    pool_output = pool_layer(conv_output)
    print(f"输入形状: {conv_output.shape}")
    print(f"输出形状: {pool_output.shape}")

    # 6. Flatten层 - 展平层
    print("\n6. Flatten层 (展平):")
    flatten_layer = tf.keras.layers.Flatten()
    flatten_output = flatten_layer(pool_output)
    print(f"输入形状: {pool_output.shape}")
    print(f"输出形状: {flatten_output.shape}")

    # 7. Embedding层 - 嵌入层 (用于文本)
    print("\n7. Embedding层 (嵌入):")
    # 假设词汇表大小为1000，嵌入维度为64
    vocab_size = 1000
    embed_dim = 64

    # 输入是整数序列 (单词索引)
    text_input = tf.constant([[1, 5, 10, 3], [2, 8, 15, 7]])  # 批次大小2, 序列长度4
    embed_layer = tf.keras.layers.Embedding(
        input_dim=vocab_size, output_dim=embed_dim, input_length=4
    )

    embed_output = embed_layer(text_input)
    print(f"输入形状: {text_input.shape}")
    print(f"输出形状: {embed_output.shape}")


# =============================
# 3. 激活函数详解
# =============================


def activation_functions():
    """
    介绍各种激活函数
    """
    print("\n" + "=" * 50)
    print("激活函数详解")
    print("=" * 50)

    # 创建测试数据
    x = tf.linspace(-5.0, 5.0, 100)
    x_np = x.numpy()

    # 定义不同的激活函数
    activations = {
        "ReLU": tf.nn.relu,
        "Sigmoid": tf.nn.sigmoid,
        "Tanh": tf.nn.tanh,
        "Leaky ReLU": tf.nn.leaky_relu,
        "ELU": tf.nn.elu,
        "Swish": tf.nn.swish,
        "Softplus": tf.nn.softplus,
    }

    print("常见激活函数及其特点:")
    for name, activation_fn in activations.items():
        y = activation_fn(x)
        print(f"\n{name}:")
        print(f"  输入范围: [-5, 5]")
        print(f"  输出范围: [{float(tf.reduce_min(y)):.2f}, {float(tf.reduce_max(y)):.2f}]")

        if name == "ReLU":
            print("  特点: 简单高效，但有神经元死亡问题")
        elif name == "Sigmoid":
            print("  特点: 输出范围(0,1)，适合二分类，但存在梯度消失")
        elif name == "Tanh":
            print("  特点: 输出范围(-1,1)，零中心化，但仍有梯度消失")
        elif name == "Leaky ReLU":
            print("  特点: 解决了ReLU的神经元死亡问题")
        elif name == "ELU":
            print("  特点: 在负值区域平滑，但计算稍复杂")
        elif name == "Swish":
            print("  特点: 自门控激活，性能优秀，但计算复杂")

    # 激活函数在模型中的使用
    print("\n激活函数在模型中的使用示例:")

    # 不同激活函数的Dense层
    model_variants = {
        "ReLU": tf.keras.Sequential(
            [
                tf.keras.layers.Dense(32, activation="relu", input_shape=(10,)),
                tf.keras.layers.Dense(16, activation="relu"),
                tf.keras.layers.Dense(1, activation="sigmoid"),
            ]
        ),
        "Tanh": tf.keras.Sequential(
            [
                tf.keras.layers.Dense(32, activation="tanh", input_shape=(10,)),
                tf.keras.layers.Dense(16, activation="tanh"),
                tf.keras.layers.Dense(1, activation="sigmoid"),
            ]
        ),
        "Leaky ReLU": tf.keras.Sequential(
            [
                tf.keras.layers.Dense(32, input_shape=(10,)),
                tf.keras.layers.LeakyReLU(alpha=0.2),
                tf.keras.layers.Dense(16),
                tf.keras.layers.LeakyReLU(alpha=0.2),
                tf.keras.layers.Dense(1, activation="sigmoid"),
            ]
        ),
    }

    sample_input = tf.random.normal((1, 10))
    for name, model in model_variants.items():
        output = model(sample_input)
        print(f"{name}模型输出: {output[0, 0].numpy():.4f}")


# =============================
# 4. 损失函数详解
# =============================


def loss_functions():
    """
    介绍各种损失函数
    """
    print("\n" + "=" * 50)
    print("损失函数详解")
    print("=" * 50)

    # 创建示例数据
    # 回归问题
    y_true_reg = tf.constant([1.0, 2.0, 3.0, 4.0])
    y_pred_reg = tf.constant([1.1, 1.9, 3.2, 3.8])

    # 分类问题
    y_true_cls = tf.constant([0, 1, 2, 1])
    y_pred_cls = tf.constant(
        [
            [0.8, 0.1, 0.1],  # 预测为类别0
            [0.1, 0.7, 0.2],  # 预测为类别1
            [0.2, 0.1, 0.7],  # 预测为类别2
            [0.3, 0.6, 0.1],  # 预测为类别1
        ]
    )

    print("1. 回归损失函数:")
    # MSE (Mean Squared Error)
    mse_loss = tf.keras.losses.MeanSquaredError()
    mse_value = mse_loss(y_true_reg, y_pred_reg)
    print(f"  MSE: {mse_value.numpy():.4f}")

    # MAE (Mean Absolute Error)
    mae_loss = tf.keras.losses.MeanAbsoluteError()
    mae_value = mae_loss(y_true_reg, y_pred_reg)
    print(f"  MAE: {mae_value.numpy():.4f}")

    # Huber Loss
    huber_loss = tf.keras.losses.Huber()
    huber_value = huber_loss(y_true_reg, y_pred_reg)
    print(f"  Huber Loss: {huber_value.numpy():.4f}")

    print("\n2. 分类损失函数:")
    # Binary Crossentropy (二分类)
    binary_true = tf.constant([0, 1, 1, 0])
    binary_pred = tf.constant([0.1, 0.9, 0.8, 0.3])

    bce_loss = tf.keras.losses.BinaryCrossentropy()
    bce_value = bce_loss(binary_true, binary_pred)
    print(f"  Binary Crossentropy: {bce_value.numpy():.4f}")

    # Categorical Crossentropy (多分类)
    one_hot_true = tf.one_hot(y_true_cls, depth=3)
    cce_loss = tf.keras.losses.CategoricalCrossentropy()
    cce_value = cce_loss(one_hot_true, y_pred_cls)
    print(f"  Categorical Crossentropy: {cce_value.numpy():.4f}")

    # Sparse Categorical Crossentropy (稀疏多分类)
    scce_loss = tf.keras.losses.SparseCategoricalCrossentropy()
    scce_value = scce_loss(y_true_cls, y_pred_cls)
    print(f"  Sparse Categorical Crossentropy: {scce_value.numpy():.4f}")

    print("\n3. 损失函数的选择指南:")
    print("  回归问题:")
    print("    - MSE: 对异常值敏感")
    print("    - MAE: 对异常值不敏感")
    print("    - Huber: MSE和MAE的折中")
    print("  分类问题:")
    print("    - BCE: 二分类问题")
    print("    - CCE: 多分类(需要one-hot编码)")
    print("    - SCCE: 多分类(直接使用类别索引)")


# =============================
# 5. 优化器详解
# =============================


def optimizers():
    """
    介绍各种优化器
    """
    print("\n" + "=" * 50)
    print("优化器详解")
    print("=" * 50)

    # 创建简单的模型和数据用于演示
    model = tf.keras.Sequential(
        [tf.keras.layers.Dense(10, activation="relu", input_shape=(5,)), tf.keras.layers.Dense(1)]
    )

    # 创建虚拟数据
    X = tf.random.normal((100, 5))
    y = tf.random.normal((100, 1))

    # 不同的优化器
    optimizers_dict = {
        "SGD": tf.keras.optimizers.SGD(learning_rate=0.01),
        "SGD + Momentum": tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9),
        "RMSprop": tf.keras.optimizers.RMSprop(learning_rate=0.001),
        "Adam": tf.keras.optimizers.Adam(learning_rate=0.001),
        "Adamax": tf.keras.optimizers.Adamax(learning_rate=0.001),
        "Nadam": tf.keras.optimizers.Nadam(learning_rate=0.001),
    }

    print("常见优化器及其特点:")
    optimizer_info = {
        "SGD": "最基础的优化器，收敛速度慢",
        "SGD + Momentum": "加入动量，加速收敛",
        "RMSprop": "自适应学习率，适合RNN",
        "Adam": "结合动量和自适应学习率，最常用",
        "Adamax": "Adam的无穷范数版本",
        "Nadam": "Adam + Nesterov动量",
    }

    for name, optimizer in optimizers_dict.items():
        # 编译模型
        model.compile(optimizer=optimizer, loss="mse")

        # 训练一个epoch
        history = model.fit(X, y, epochs=1, verbose=0)
        loss = history.history["loss"][0]

        print(f"{name}:")
        print(f"  1轮训练后损失: {loss:.4f}")
        print(f"  特点: {optimizer_info[name]}")

    print("\n优化器使用建议:")
    print("  - 初学者: 推荐使用Adam，默认参数通常效果不错")
    print("  - 追求性能: 可以尝试Adam、Nadam或调整参数")
    print("  - 简单问题: SGD + Momentum可能足够")
    print("  - RNN任务: RMSprop是不错的选择")


# =============================
# 6. 简单神经网络训练示例
# =============================


def simple_neural_network_example():
    """
    构建和训练一个简单的神经网络
    """
    print("\n" + "=" * 50)
    print("简单神经网络训练示例")
    print("=" * 50)

    # 1. 准备数据
    print("1. 准备数据:")
    # 生成模拟数据
    X, y = make_moons(n_samples=1000, noise=0.2, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    print(f"训练集大小: {X_train.shape}")
    print(f"测试集大小: {X_test.shape}")

    # 2. 构建模型
    print("\n2. 构建模型:")
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Dense(16, activation="relu", input_shape=(2,)),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(8, activation="relu"),
            tf.keras.layers.Dense(1, activation="sigmoid"),
        ]
    )

    model.summary()

    # 3. 编译模型
    print("\n3. 编译模型:")
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

    # 4. 训练模型
    print("\n4. 训练模型:")
    history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2, verbose=0)

    # 5. 评估模型
    print("\n5. 评估模型:")
    train_loss, train_acc = model.evaluate(X_train, y_train, verbose=0)
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)

    print(f"训练集 - 损失: {train_loss:.4f}, 准确率: {train_acc:.4f}")
    print(f"测试集 - 损失: {test_loss:.4f}, 准确率: {test_acc:.4f}")

    # 6. 预测
    print("\n6. 预测示例:")
    samples = X_test[:5]
    predictions = model.predict(samples)
    predicted_classes = (predictions > 0.5).astype(int).flatten()

    print("真实标签:", y_test[:5])
    print("预测概率:", predictions.flatten())
    print("预测类别:", predicted_classes)


# =============================
# 主函数
# =============================


def main():
    """
    主函数，执行所有神经网络基础示例
    """
    print("TensorFlow 2.0 神经网络基础学习")
    print("=" * 60)

    # 执行各个模块
    keras_api_introduction()
    neural_network_layers()
    activation_functions()
    loss_functions()
    optimizers()
    simple_neural_network_example()

    print("\n" + "=" * 60)
    print("神经网络基础学习完成！")


if __name__ == "__main__":
    main()
