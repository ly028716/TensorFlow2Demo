"""
TensorFlow 2.0 自定义层和模型

本模块介绍如何创建自定义神经网络层和模型
包括继承基类实现自定义功能
"""

import tensorflow as tf
import numpy as np


# =============================
# 1. 自定义层
# =============================


class CustomDense(tf.keras.layers.Layer):
    """
    自定义全连接层
    演示如何实现一个简单的Dense层
    """

    def __init__(self, units=32, **kwargs):
        super(CustomDense, self).__init__(**kwargs)
        self.units = units

    def build(self, input_shape):
        """构建层的权重"""
        # 权重矩阵: [input_dim, units]
        self.w = self.add_weight(
            shape=(input_shape[-1], self.units),
            initializer="glorot_uniform",
            trainable=True,
            name="kernel",
        )
        # 偏置向量: [units]
        self.b = self.add_weight(
            shape=(self.units,), initializer="zeros", trainable=True, name="bias"
        )
        super(CustomDense, self).build(input_shape)

    def call(self, inputs):
        """前向传播"""
        # 线性变换: x @ w + b
        return tf.matmul(inputs, self.w) + self.b

    def get_config(self):
        """获取层的配置，用于模型保存和加载"""
        config = super(CustomDense, self).get_config()
        config.update({"units": self.units})
        return config


class CustomDropout(tf.keras.layers.Layer):
    """
    自定义Dropout层
    演示如何实现正则化层
    """

    def __init__(self, rate, **kwargs):
        super(CustomDropout, self).__init__(**kwargs)
        self.rate = rate

    def call(self, inputs, training=None):
        """前向传播"""
        if training:
            # 训练时: 随机将部分元素置零
            mask = tf.cast(tf.random.uniform(tf.shape(inputs)) > self.rate, inputs.dtype)
            return inputs * mask / (1 - self.rate)
        else:
            # 推理时: 直接返回输入
            return inputs

    def get_config(self):
        """获取层的配置"""
        config = super(CustomDropout, self).get_config()
        config.update({"rate": self.rate})
        return config


class CustomActivation(tf.keras.layers.Layer):
    """
    自定义激活函数层
    演示如何实现Swish激活函数
    """

    def __init__(self, **kwargs):
        super(CustomActivation, self).__init__(**kwargs)

    def call(self, inputs):
        """Swish激活函数: x * sigmoid(x)"""
        return inputs * tf.nn.sigmoid(inputs)


class CustomResidualBlock(tf.keras.layers.Layer):
    """
    自定义残差块
    演示如何实现复杂的层组合
    """

    def __init__(self, filters, stride=1, **kwargs):
        super(CustomResidualBlock, self).__init__(**kwargs)
        self.filters = filters
        self.stride = stride

        # 第一个卷积层
        self.conv1 = tf.keras.layers.Conv2D(
            filters, kernel_size=3, strides=stride, padding="same", use_bias=False
        )
        self.bn1 = tf.keras.layers.BatchNormalization()

        # 第二个卷积层
        self.conv2 = tf.keras.layers.Conv2D(
            filters, kernel_size=3, strides=1, padding="same", use_bias=False
        )
        self.bn2 = tf.keras.layers.BatchNormalization()

        # 残差连接的投影层(当维度不匹配时使用)
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
        """前向传播"""
        residual = self.shortcut(inputs)

        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = tf.nn.relu(x)

        x = self.conv2(x)
        x = self.bn2(x, training=training)

        # 残差连接
        x += residual
        return tf.nn.relu(x)


# =============================
# 2. 自定义模型
# =============================


class CustomMLP(tf.keras.Model):
    """
    自定义多层感知机模型
    演示基本的模型子类化
    """

    def __init__(self, hidden_units=[64, 32], output_units=10, **kwargs):
        super(CustomMLP, self).__init__(**kwargs)

        self.hidden_layers = []
        for units in hidden_units:
            self.hidden_layers.extend([CustomDense(units), CustomActivation(), CustomDropout(0.2)])

        self.output_layer = CustomDense(output_units)
        self.softmax = tf.keras.layers.Softmax()

    def call(self, inputs, training=None):
        """前向传播"""
        x = inputs
        for layer in self.hidden_layers:
            if isinstance(layer, CustomDropout):
                x = layer(x, training=training)
            else:
                x = layer(x)

        x = self.output_layer(x)
        return self.softmax(x)


class CustomCNN(tf.keras.Model):
    """
    自定义卷积神经网络模型
    演示CNN模型的实现
    """

    def __init__(self, num_classes=10, **kwargs):
        super(CustomCNN, self).__init__(**kwargs)

        # 特征提取部分
        self.feature_extractor = tf.keras.Sequential(
            [
                tf.keras.layers.Conv2D(32, kernel_size=3, activation="relu", padding="same"),
                tf.keras.layers.MaxPooling2D(pool_size=2),
                CustomResidualBlock(32),
                tf.keras.layers.Conv2D(64, kernel_size=3, activation="relu", padding="same"),
                tf.keras.layers.MaxPooling2D(pool_size=2),
                CustomResidualBlock(64),
                tf.keras.layers.Conv2D(128, kernel_size=3, activation="relu", padding="same"),
                tf.keras.layers.GlobalAveragePooling2D(),
            ]
        )

        # 分类部分
        self.classifier = tf.keras.Sequential(
            [CustomDense(128), CustomActivation(), CustomDropout(0.5), CustomDense(num_classes)]
        )

    def call(self, inputs, training=None):
        """前向传播"""
        x = self.feature_extractor(inputs, training=training)
        return self.classifier(x, training=training)


class CustomAutoencoder(tf.keras.Model):
    """
    自定义自编码器模型
    演示编码器-解码器结构
    """

    def __init__(self, latent_dim=32, **kwargs):
        super(CustomAutoencoder, self).__init__(**kwargs)
        self.latent_dim = latent_dim

        # 编码器
        self.encoder = tf.keras.Sequential(
            [
                CustomDense(128, name="encoder_1"),
                CustomActivation(),
                CustomDense(64, name="encoder_2"),
                CustomActivation(),
                CustomDense(latent_dim, name="latent"),
            ]
        )

        # 解码器
        self.decoder = tf.keras.Sequential(
            [
                CustomDense(64, name="decoder_1"),
                CustomActivation(),
                CustomDense(128, name="decoder_2"),
                CustomActivation(),
                CustomDense(784, name="reconstruction"),  # 28*28=784 (MNIST)
            ]
        )

    def call(self, inputs):
        """前向传播"""
        encoded = self.encoder(inputs)
        decoded = self.decoder(encoded)
        return decoded

    def encode(self, inputs):
        """编码函数"""
        return self.encoder(inputs)

    def decode(self, latent):
        """解码函数"""
        return self.decoder(latent)


# =============================
# 3. 自定义训练循环
# =============================


class CustomTrainer:
    """
    自定义训练循环类
    演示如何实现完整的训练过程
    """

    def __init__(self, model, optimizer, loss_fn):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.train_loss = tf.keras.metrics.Mean(name="train_loss")
        self.train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name="train_accuracy")
        self.val_loss = tf.keras.metrics.Mean(name="val_loss")
        self.val_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name="val_accuracy")

    @tf.function
    def train_step(self, images, labels):
        """单个训练步骤"""
        with tf.GradientTape() as tape:
            predictions = self.model(images, training=True)
            loss = self.loss_fn(labels, predictions)

        # 计算梯度并更新权重
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

        # 更新指标
        self.train_loss(loss)
        self.train_accuracy(labels, predictions)

    @tf.function
    def val_step(self, images, labels):
        """单个验证步骤"""
        predictions = self.model(images, training=False)
        loss = self.loss_fn(labels, predictions)

        # 更新指标
        self.val_loss(loss)
        self.val_accuracy(labels, predictions)

    def train(self, train_ds, val_ds, epochs):
        """完整训练循环"""
        history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

        for epoch in range(epochs):
            print(f"Epoch {epoch + 1}/{epochs}")

            # 重置指标
            self.train_loss.reset_states()
            self.train_accuracy.reset_states()
            self.val_loss.reset_states()
            self.val_accuracy.reset_states()

            # 训练阶段
            for images, labels in train_ds:
                self.train_step(images, labels)

            # 验证阶段
            for images, labels in val_ds:
                self.val_step(images, labels)

            # 记录历史
            history["train_loss"].append(self.train_loss.result())
            history["train_acc"].append(self.train_accuracy.result())
            history["val_loss"].append(self.val_loss.result())
            history["val_acc"].append(self.val_accuracy.result())

            # 打印进度
            template = "Loss: {:.4f}, Accuracy: {:.4f}, Val Loss: {:.4f}, Val Accuracy: {:.4f}"
            print(
                template.format(
                    self.train_loss.result(),
                    self.train_accuracy.result() * 100,
                    self.val_loss.result(),
                    self.val_accuracy.result() * 100,
                )
            )

        return history


# =============================
# 4. 示例和演示
# =============================


def custom_layers_demo():
    """
    演示自定义层的功能
    """
    print("=" * 50)
    print("自定义层演示")
    print("=" * 50)

    # 创建测试数据
    inputs = tf.random.normal((2, 10))

    # 测试自定义Dense层
    print("1. 自定义Dense层:")
    custom_dense = CustomDense(units=5)
    output = custom_dense(inputs)
    print(f"输入形状: {inputs.shape}")
    print(f"输出形状: {output.shape}")
    print(f"权重形状: {custom_dense.w.shape}")
    print(f"偏置形状: {custom_dense.b.shape}")

    # 测试自定义Dropout层
    print("\n2. 自定义Dropout层:")
    custom_dropout = CustomDropout(rate=0.5)
    output_train = custom_dropout(inputs, training=True)
    output_test = custom_dropout(inputs, training=False)
    print(f"训练时输出: {output_train[0, :3].numpy()}")
    print(f"测试时输出: {output_test[0, :3].numpy()}")

    # 测试自定义激活函数
    print("\n3. 自定义激活函数:")
    custom_activation = CustomActivation()
    output_activation = custom_activation(inputs)
    print(f"输入: {inputs[0, :3].numpy()}")
    print(f"Swish激活输出: {output_activation[0, :3].numpy()}")

    # 测试自定义残差块
    print("\n4. 自定义残差块:")
    image_input = tf.random.normal((2, 28, 28, 3))
    residual_block = CustomResidualBlock(filters=32)
    output_residual = residual_block(image_input)
    print(f"输入形状: {image_input.shape}")
    print(f"输出形状: {output_residual.shape}")


def custom_models_demo():
    """
    演示自定义模型的功能
    """
    print("\n" + "=" * 50)
    print("自定义模型演示")
    print("=" * 50)

    # 创建不同类型的自定义模型
    print("1. 自定义MLP:")
    mlp_model = CustomMLP(hidden_units=[64, 32], output_units=10)
    mlp_input = tf.random.normal((1, 20))
    mlp_output = mlp_model(mlp_input)
    print(f"输入形状: {mlp_input.shape}")
    print(f"输出形状: {mlp_output.shape}")

    print("\n2. 自定义CNN:")
    cnn_model = CustomCNN(num_classes=10)
    cnn_input = tf.random.normal((1, 32, 32, 3))
    cnn_output = cnn_model(cnn_input)
    print(f"输入形状: {cnn_input.shape}")
    print(f"输出形状: {cnn_output.shape}")

    print("\n3. 自定义自编码器:")
    autoencoder = CustomAutoencoder(latent_dim=16)
    ae_input = tf.random.normal((1, 784))
    ae_output = autoencoder(ae_input)
    encoded = autoencoder.encode(ae_input)
    print(f"输入形状: {ae_input.shape}")
    print(f"潜在表示形状: {encoded.shape}")
    print(f"重构输出形状: {ae_output.shape}")


def custom_training_demo():
    """
    演示自定义训练循环
    """
    print("\n" + "=" * 50)
    print("自定义训练循环演示")
    print("=" * 50)

    # 创建简单的分类任务
    num_samples = 1000
    num_features = 20
    num_classes = 5

    # 生成虚拟数据
    X = tf.random.normal((num_samples, num_features))
    y = tf.random.uniform((num_samples,), maxval=num_classes, dtype=tf.int32)

    # 创建数据集
    dataset = tf.data.Dataset.from_tensor_slices((X, y))
    dataset = dataset.shuffle(buffer_size=1000).batch(32)

    # 分割训练和验证集
    train_size = int(0.8 * num_samples)
    train_dataset = dataset.take(train_size)
    val_dataset = dataset.skip(train_size)

    # 创建模型和训练器
    model = CustomMLP(hidden_units=[64, 32], output_units=num_classes)
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()
    trainer = CustomTrainer(model, optimizer, loss_fn)

    print("开始自定义训练...")
    history = trainer.train(train_dataset, val_dataset, epochs=3)

    print("\n训练完成!")
    print("最终训练准确率:", history["train_acc"][-1].numpy())
    print("最终验证准确率:", history["val_acc"][-1].numpy())


# =============================
# 主函数
# =============================


def main():
    """
    主函数，执行所有自定义层和模型示例
    """
    print("TensorFlow 2.0 自定义层和模型学习")
    print("=" * 60)

    # 执行各个模块
    custom_layers_demo()
    custom_models_demo()
    custom_training_demo()

    print("\n" + "=" * 60)
    print("自定义层和模型学习完成！")
    print("\n关键要点:")
    print("1. 继承tf.keras.layers.Layer创建自定义层")
    print("2. 继承tf.keras.Model创建自定义模型")
    print("3. 实现build()方法初始化权重")
    print("4. 实现call()方法定义前向传播")
    print("5. 实现get_config()方法支持序列化")
    print("6. 可以实现自定义训练循环以获得更多控制")


if __name__ == "__main__":
    main()
