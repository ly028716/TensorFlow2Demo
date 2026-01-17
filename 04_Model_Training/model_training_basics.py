"""
TensorFlow 2.0 模型训练与评估基础

本模块介绍TensorFlow 2.0中的模型训练和评估技术
包括编译、训练、验证、保存和加载模型
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split


# =============================
# 1. 模型编译与训练基础
# =============================


def model_compilation_basics():
    """
    模型编译基础
    """
    print("=" * 50)
    print("模型编译基础")
    print("=" * 50)

    # 创建简单模型
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Dense(32, activation="relu", input_shape=(10,)),
            tf.keras.layers.Dense(16, activation="relu"),
            tf.keras.layers.Dense(1, activation="sigmoid"),
        ]
    )

    # 1. 基本编译
    print("1. 基本编译:")
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    print("模型已使用默认参数编译")

    # 2. 使用特定的优化器和损失函数
    print("\n2. 使用特定的优化器和损失函数:")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=[
            tf.keras.metrics.BinaryAccuracy(name="accuracy"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
        ],
    )
    print("模型已使用自定义优化器和指标编译")

    # 3. 显示模型配置
    print("\n3. 模型配置:")
    print(f"优化器: {model.optimizer}")
    print(f"损失函数: {model.loss}")
    print(f"指标: {[m.name for m in model.metrics]}")
    print(f"可训练参数数量: {model.count_params()}")


def basic_model_training():
    """
    基本模型训练
    """
    print("\n" + "=" * 50)
    print("基本模型训练")
    print("=" * 50)

    # 创建分类数据
    X, y = make_classification(
        n_samples=1000, n_features=20, n_classes=2, n_redundant=5, random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    print(f"训练集大小: {X_train.shape}")
    print(f"验证集大小: {X_val.shape}")

    # 创建模型
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Dense(64, activation="relu", input_shape=(20,)),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(32, activation="relu"),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(1, activation="sigmoid"),
        ]
    )

    # 编译模型
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

    # 训练模型
    print("\n开始训练...")
    history = model.fit(
        X_train, y_train, epochs=20, batch_size=32, validation_data=(X_val, y_val), verbose=1
    )

    # 打印训练历史
    print("\n训练完成!")
    final_train_acc = history.history["accuracy"][-1]
    final_val_acc = history.history["val_accuracy"][-1]
    print(f"最终训练准确率: {final_train_acc:.4f}")
    print(f"最终验证准确率: {final_val_acc:.4f}")

    return model, history


# =============================
# 2. 训练回调函数
# =============================


def training_callbacks_demo():
    """
    训练回调函数演示
    """
    print("\n" + "=" * 50)
    print("训练回调函数演示")
    print("=" * 50)

    # 创建数据
    X, y = make_classification(
        n_samples=1000, n_features=20, n_classes=2, n_redundant=5, random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # 创建模型
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Dense(64, activation="relu", input_shape=(20,)),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(32, activation="relu"),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(1, activation="sigmoid"),
        ]
    )

    # 编译模型
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

    # 1. EarlyStopping回调
    print("1. EarlyStopping回调:")
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=5, restore_best_weights=True, verbose=1
    )

    # 2. ModelCheckpoint回调
    print("2. ModelCheckpoint回调:")
    checkpoint_dir = "checkpoints"
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(checkpoint_dir, "best_model.h5"),
        monitor="val_accuracy",
        save_best_only=True,
        save_weights_only=False,
        verbose=1,
    )

    # 3. ReduceLROnPlateau回调
    print("3. ReduceLROnPlateau回调:")
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.2, patience=3, min_lr=0.0001, verbose=1
    )

    # 4. TensorBoard回调
    print("4. TensorBoard回调:")
    log_dir = "logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    tensorboard = tf.keras.callbacks.TensorBoard(
        log_dir=log_dir, histogram_freq=1, write_graph=True, write_images=True
    )

    # 5. 自定义回调
    print("5. 自定义回调:")

    class CustomCallback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            if epoch % 5 == 0:
                print(f"Epoch {epoch}: Custom callback - Loss = {logs['loss']:.4f}")

        def on_train_begin(self, logs=None):
            print("Training started!")

        def on_train_end(self, logs=None):
            print("Training completed!")

    custom_callback = CustomCallback()

    # 训练模型，使用回调函数
    callbacks = [early_stopping, model_checkpoint, reduce_lr, tensorboard, custom_callback]

    print("\n开始使用回调函数训练...")
    history = model.fit(
        X_train,
        y_train,
        epochs=50,
        batch_size=32,
        validation_data=(X_val, y_val),
        callbacks=callbacks,
        verbose=2,
    )

    print(f"训练轮数: {len(history.history['loss'])}")
    print(f"最佳验证准确率: {max(history.history['val_accuracy']):.4f}")

    return model, history


# =============================
# 3. 自定义训练循环
# =============================


def custom_training_loop():
    """
    自定义训练循环
    """
    print("\n" + "=" * 50)
    print("自定义训练循环")
    print("=" * 50)

    # 创建数据
    X, y = make_classification(
        n_samples=1000, n_features=20, n_classes=2, n_redundant=5, random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # 创建数据集
    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    train_dataset = train_dataset.shuffle(buffer_size=len(X_train)).batch(32)

    val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val))
    val_dataset = val_dataset.batch(32)

    # 创建模型
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Dense(64, activation="relu", input_shape=(20,)),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(32, activation="relu"),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(1, activation="sigmoid"),
        ]
    )

    # 定义优化器和损失函数
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    loss_fn = tf.keras.losses.BinaryCrossentropy()
    train_accuracy = tf.keras.metrics.BinaryAccuracy()
    val_accuracy = tf.keras.metrics.BinaryAccuracy()
    train_loss = tf.keras.metrics.Mean()
    val_loss = tf.keras.metrics.Mean()

    # 训练参数
    epochs = 20

    print("开始自定义训练循环...")
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")

        # 训练阶段
        train_accuracy.reset_states()
        train_loss.reset_states()

        for step, (x_batch, y_batch) in enumerate(train_dataset):
            with tf.GradientTape() as tape:
                predictions = model(x_batch, training=True)
                loss = loss_fn(y_batch, predictions)

            # 计算梯度并更新权重
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))

            # 更新指标
            train_loss.update_state(loss)
            train_accuracy.update_state(y_batch, predictions)

            # 打印进度
            if step % 10 == 0:
                print(f"  Step {step}, Loss: {loss:.4f}")

        # 验证阶段
        val_accuracy.reset_states()
        val_loss.reset_states()

        for x_batch_val, y_batch_val in val_dataset:
            val_predictions = model(x_batch_val, training=False)
            val_loss_value = loss_fn(y_batch_val, val_predictions)
            val_loss.update_state(val_loss_value)
            val_accuracy.update_state(y_batch_val, val_predictions)

        # 打印每轮的结果
        print(f"  Train Loss: {train_loss.result():.4f}, Train Acc: {train_accuracy.result():.4f}")
        print(f"  Val Loss: {val_loss.result():.4f}, Val Acc: {val_accuracy.result():.4f}")

    print("自定义训练循环完成!")
    return model


# =============================
# 4. 模型评估
# =============================


def model_evaluation():
    """
    模型评估演示
    """
    print("\n" + "=" * 50)
    print("模型评估")
    print("=" * 50)

    # 创建数据
    X, y = make_classification(
        n_samples=1000, n_features=20, n_classes=2, n_redundant=5, random_state=42
    )
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # 创建并训练模型
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Dense(64, activation="relu", input_shape=(20,)),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(32, activation="relu"),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(1, activation="sigmoid"),
        ]
    )

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.Precision(), tf.keras.metrics.Recall()],
    )

    # 训练模型
    print("训练模型...")
    model.fit(X_train, y_train, epochs=20, batch_size=32, verbose=0)

    # 1. 基本评估
    print("\n1. 基本评估:")
    test_loss, test_acc, test_precision, test_recall = model.evaluate(X_test, y_test, verbose=0)
    print(f"测试损失: {test_loss:.4f}")
    print(f"测试准确率: {test_acc:.4f}")
    print(f"测试精确率: {test_precision:.4f}")
    print(f"测试召回率: {test_recall:.4f}")

    # 2. 预测
    print("\n2. 预测:")
    predictions = model.predict(X_test[:5], verbose=0)
    predicted_classes = (predictions > 0.5).astype(int).flatten()
    true_classes = y_test[:5]

    print("前5个样本的预测结果:")
    for i in range(5):
        print(
            f"  样本{i+1}: 真实={true_classes[i]}, 预测概率={predictions[i][0]:.4f}, 预测类别={predicted_classes[i]}"
        )

    # 3. 混淆矩阵
    print("\n3. 混淆矩阵:")
    all_predictions = model.predict(X_test, verbose=0)
    all_predicted_classes = (all_predictions > 0.5).astype(int).flatten()

    # 计算混淆矩阵
    tp = np.sum((all_predicted_classes == 1) & (y_test == 1))
    tn = np.sum((all_predicted_classes == 0) & (y_test == 0))
    fp = np.sum((all_predicted_classes == 1) & (y_test == 0))
    fn = np.sum((all_predicted_classes == 0) & (y_test == 1))

    print(f"真阳性 (TP): {tp}")
    print(f"真阴性 (TN): {tn}")
    print(f"假阳性 (FP): {fp}")
    print(f"假阴性 (FN): {fn}")

    # 计算F1分数
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    print(f"\n精确率: {precision:.4f}")
    print(f"召回率: {recall:.4f}")
    print(f"F1分数: {f1_score:.4f}")

    return model


# =============================
# 5. 模型保存与加载
# =============================


def model_saving_loading():
    """
    模型保存与加载
    """
    print("\n" + "=" * 50)
    print("模型保存与加载")
    print("=" * 50)

    # 创建模型
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Dense(64, activation="relu", input_shape=(20,)),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(32, activation="relu"),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(1, activation="sigmoid"),
        ]
    )

    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

    # 训练模型
    X, y = make_classification(n_samples=100, n_features=20, n_classes=2, random_state=42)
    model.fit(X, y, epochs=5, batch_size=16, verbose=0)

    # 1. 保存整个模型
    print("1. 保存整个模型:")
    model.save("full_model.h5")
    print("模型已保存为 'full_model.h5'")

    # 加载模型
    loaded_model = tf.keras.models.load_model("full_model.h5")
    print("模型已加载")
    print(f"原始模型预测: {model.predict(X[:1], verbose=0)[0][0]:.4f}")
    print(f"加载模型预测: {loaded_model.predict(X[:1], verbose=0)[0][0]:.4f}")

    # 2. 只保存权重
    print("\n2. 只保存权重:")
    model.save_weights("model_weights.h5")
    print("权重已保存为 'model_weights.h5'")

    # 创建相同架构的新模型并加载权重
    new_model = tf.keras.Sequential(
        [
            tf.keras.layers.Dense(64, activation="relu", input_shape=(20,)),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(32, activation="relu"),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(1, activation="sigmoid"),
        ]
    )

    new_model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

    new_model.load_weights("model_weights.h5")
    print("权重已加载到新模型")
    print(f"新模型预测: {new_model.predict(X[:1], verbose=0)[0][0]:.4f}")

    # 3. 保存模型架构 (JSON格式)
    print("\n3. 保存模型架构:")
    model_json = model.to_json()
    with open("model_architecture.json", "w") as f:
        f.write(model_json)
    print("模型架构已保存为 'model_architecture.json'")

    # 从JSON加载架构
    with open("model_architecture.json", "r") as f:
        loaded_model_json = f.read()
    model_from_json = tf.keras.models.model_from_json(loaded_model_json)
    model_from_json.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    print("模型架构已从JSON加载")

    # 4. SavedModel格式保存
    print("\n4. SavedModel格式保存:")
    model.save("saved_model/")
    print("模型已保存为SavedModel格式")

    # 从SavedModel加载
    restored_model = tf.keras.models.load_model("saved_model/")
    print("模型已从SavedFormat加载")
    print(f"恢复模型预测: {restored_model.predict(X[:1], verbose=0)[0][0]:.4f}")


# =============================
# 6. 可视化训练过程
# =============================


def visualize_training():
    """
    可视化训练过程
    """
    print("\n" + "=" * 50)
    print("可视化训练过程")
    print("=" * 50)

    # 创建数据
    X, y = make_classification(
        n_samples=1000, n_features=20, n_classes=2, n_redundant=5, random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # 创建模型
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Dense(64, activation="relu", input_shape=(20,)),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(32, activation="relu"),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(1, activation="sigmoid"),
        ]
    )

    # 编译模型
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

    # 训练模型
    print("训练模型并记录历史...")
    history = model.fit(
        X_train, y_train, epochs=30, batch_size=32, validation_data=(X_val, y_val), verbose=0
    )

    # 1. 绘制损失曲线
    print("1. 绘制损失曲线:")
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(history.history["loss"], label="训练损失")
    plt.plot(history.history["val_loss"], label="验证损失")
    plt.title("模型损失")
    plt.xlabel("Epoch")
    plt.ylabel("损失")
    plt.legend()
    plt.grid(True)

    # 2. 绘制准确率曲线
    print("2. 绘制准确率曲线:")
    plt.subplot(1, 2, 2)
    plt.plot(history.history["accuracy"], label="训练准确率")
    plt.plot(history.history["val_accuracy"], label="验证准确率")
    plt.title("模型准确率")
    plt.xlabel("Epoch")
    plt.ylabel("准确率")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig("training_history.png")
    print("训练历史图表已保存为 'training_history.png'")

    # 3. 打印关键信息
    print("\n3. 关键训练信息:")
    best_epoch = np.argmax(history.history["val_accuracy"]) + 1
    best_val_acc = max(history.history["val_accuracy"])
    final_val_acc = history.history["val_accuracy"][-1]

    print(f"最佳验证准确率: {best_val_acc:.4f} (第{best_epoch}轮)")
    print(f"最终验证准确率: {final_val_acc:.4f}")

    # 检查过拟合
    if final_val_acc < best_val_acc * 0.95:  # 如果最终值比最佳值下降5%以上
        print("模型可能存在过拟合，建议早停或增加正则化")
    else:
        print("模型训练稳定")

    return history


# =============================
# 主函数
# =============================


def main():
    """
    主函数，执行所有模型训练和评估示例
    """
    print("TensorFlow 2.0 模型训练与评估基础学习")
    print("=" * 60)

    # 执行各个模块
    model_compilation_basics()
    basic_model_training()
    training_callbacks_demo()
    custom_training_loop()
    model_evaluation()
    model_saving_loading()
    visualize_training()

    print("\n" + "=" * 60)
    print("模型训练与评估基础学习完成！")
    print("\n关键要点:")
    print("1. 编译模型时需要指定优化器、损失函数和评估指标")
    print("2. 回调函数可以在训练过程中执行自定义操作")
    print("3. 自定义训练循环提供更多控制权")
    print("4. 合适的评估指标对模型性能分析至关重要")
    print("5. 保存和加载模型对于长期开发很重要")
    print("6. 可视化训练过程有助于理解模型行为")


if __name__ == "__main__":
    main()
