"""
TensorFlow 2.0 文本分类实用案例

本模块展示如何使用TensorFlow 2.0构建和训练文本分类模型
包括词嵌入、RNN、Transformer等技术
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


# =============================
# 1. 数据准备和预处理
# =============================


def prepare_imdb_data():
    """
    准备IMDB电影评论数据集
    """
    print("=" * 50)
    print("准备IMDB电影评论数据集")
    print("=" * 50)

    # 加载IMDB数据集
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.imdb.load_data(num_words=10000)

    print(f"训练样本数: {len(x_train)}")
    print(f"测试样本数: {len(x_test)}")
    print(f"示例训练序列长度: {len(x_train[0])}")

    # 获取词到索引的映射
    word_index = tf.keras.datasets.imdb.get_word_index()
    word_index = {k: (v + 3) for k, v in word_index.items()}
    word_index["<PAD>"] = 0
    word_index["<START>"] = 1
    word_index["<UNK>"] = 2
    word_index["<UNUSED>"] = 3

    reverse_word_index = {v: k for k, v in word_index.items()}

    def decode_review(text):
        """将数字序列解码为文本"""
        return " ".join([reverse_word_index.get(i, "?") for i in text])

    # 打印示例评论
    print("\n示例评论:")
    print(decode_review(x_train[0]))
    print(f"情感标签: {'正面' if y_train[0] == 1 else '负面'}")

    # 填充序列到相同长度
    max_length = 256
    x_train = pad_sequences(x_train, maxlen=max_length, padding="post", truncating="post")
    x_test = pad_sequences(x_test, maxlen=max_length, padding="post", truncating="post")

    print(f"\n填充后训练数据形状: {x_train.shape}")
    print(f"填充后测试数据形状: {x_test.shape}")

    return (x_train, y_train), (x_test, y_test), word_index


# =============================
# 2. 使用预训练词嵌入的文本分类
# =============================


def text_classification_with_embeddings(x_train, y_train, x_test, y_test, word_index):
    """
    使用预训练词嵌入的文本分类
    """
    print("\n" + "=" * 50)
    print("使用预训练词嵌入的文本分类")
    print("=" * 50)

    # 分割验证集
    x_val = x_train[:5000]
    y_val = y_train[:5000]
    x_train = x_train[5000:]
    y_train = y_train[5000:]

    # 1. 使用随机初始化的嵌入层
    print("1. 使用随机初始化的嵌入层:")
    embedding_dim = 100
    vocab_size = len(word_index) + 1

    model_random = tf.keras.Sequential(
        [
            tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=256),
            tf.keras.layers.GlobalAveragePooling1D(),
            tf.keras.layers.Dense(16, activation="relu"),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(1, activation="sigmoid"),
        ]
    )

    model_random.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

    print("随机嵌入模型:")
    model_random.summary()

    # 训练随机嵌入模型
    print("\n训练随机嵌入模型...")
    history_random = model_random.fit(
        x_train, y_train, epochs=20, batch_size=512, validation_data=(x_val, y_val), verbose=1
    )

    # 2. 使用GloVe预训练嵌入 (模拟)
    print("\n2. 使用GloVe预训练嵌入 (模拟):")
    # 创建模拟的预训练嵌入矩阵
    embedding_matrix = np.random.normal(0, 0.1, (vocab_size, embedding_dim))

    model_pretrained = tf.keras.Sequential(
        [
            tf.keras.layers.Embedding(
                vocab_size,
                embedding_dim,
                weights=[embedding_matrix],
                input_length=256,
                trainable=False,  # 冻结嵌入层
            ),
            tf.keras.layers.GlobalAveragePooling1D(),
            tf.keras.layers.Dense(16, activation="relu"),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(1, activation="sigmoid"),
        ]
    )

    model_pretrained.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

    print("预训练嵌入模型:")
    model_pretrained.summary()

    # 训练预训练嵌入模型
    print("\n训练预训练嵌入模型...")
    history_pretrained = model_pretrained.fit(
        x_train, y_train, epochs=20, batch_size=512, validation_data=(x_val, y_val), verbose=1
    )

    # 3. 可训练的预训练嵌入 (微调)
    print("\n3. 可训练的预训练嵌入 (微调):")
    model_finetune = tf.keras.Sequential(
        [
            tf.keras.layers.Embedding(
                vocab_size,
                embedding_dim,
                weights=[embedding_matrix],
                input_length=256,
                trainable=True,  # 允许微调嵌入层
            ),
            tf.keras.layers.GlobalAveragePooling1D(),
            tf.keras.layers.Dense(16, activation="relu"),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(1, activation="sigmoid"),
        ]
    )

    model_finetune.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

    print("微调嵌入模型:")
    model_finetune.summary()

    # 训练微调嵌入模型
    print("\n训练微调嵌入模型...")
    history_finetune = model_finetune.fit(
        x_train, y_train, epochs=20, batch_size=512, validation_data=(x_val, y_val), verbose=1
    )

    # 评估所有模型
    print("\n模型评估:")
    models = [
        ("随机嵌入", model_random, history_random),
        ("预训练嵌入", model_pretrained, history_pretrained),
        ("微调嵌入", model_finetune, history_finetune),
    ]

    for name, model, history in models:
        test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
        print(f"{name}: 测试准确率 = {test_acc:.4f}")

    return models


# =============================
# 3. 使用RNN/LSTM的文本分类
# =============================


def text_classification_with_rnn(x_train, y_train, x_test, y_test, word_index):
    """
    使用RNN/LSTM的文本分类
    """
    print("\n" + "=" * 50)
    print("使用RNN/LSTM的文本分类")
    print("=" * 50)

    # 分割验证集
    x_val = x_train[:5000]
    y_val = y_train[:5000]
    x_train = x_train[5000:]
    y_train = y_train[5000:]

    vocab_size = len(word_index) + 1
    embedding_dim = 100

    # 1. 简单RNN模型
    print("1. 简单RNN模型:")
    model_rnn = tf.keras.Sequential(
        [
            tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=256),
            tf.keras.layers.SimpleRNN(64),
            tf.keras.layers.Dense(16, activation="relu"),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(1, activation="sigmoid"),
        ]
    )

    model_rnn.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

    print("RNN模型:")
    model_rnn.summary()

    # 训练RNN模型
    print("\n训练RNN模型...")
    history_rnn = model_rnn.fit(
        x_train, y_train, epochs=10, batch_size=256, validation_data=(x_val, y_val), verbose=1
    )

    # 2. LSTM模型
    print("\n2. LSTM模型:")
    model_lstm = tf.keras.Sequential(
        [
            tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=256),
            tf.keras.layers.LSTM(64),
            tf.keras.layers.Dense(16, activation="relu"),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(1, activation="sigmoid"),
        ]
    )

    model_lstm.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

    print("LSTM模型:")
    model_lstm.summary()

    # 训练LSTM模型
    print("\n训练LSTM模型...")
    history_lstm = model_lstm.fit(
        x_train, y_train, epochs=10, batch_size=256, validation_data=(x_val, y_val), verbose=1
    )

    # 3. 双向LSTM模型
    print("\n3. 双向LSTM模型:")
    model_bilstm = tf.keras.Sequential(
        [
            tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=256),
            tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
            tf.keras.layers.Dense(16, activation="relu"),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(1, activation="sigmoid"),
        ]
    )

    model_bilstm.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

    print("双向LSTM模型:")
    model_bilstm.summary()

    # 训练双向LSTM模型
    print("\n训练双向LSTM模型...")
    history_bilstm = model_bilstm.fit(
        x_train, y_train, epochs=10, batch_size=256, validation_data=(x_val, y_val), verbose=1
    )

    # 4. 堆叠LSTM模型
    print("\n4. 堆叠LSTM模型:")
    model_stacked_lstm = tf.keras.Sequential(
        [
            tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=256),
            tf.keras.layers.LSTM(64, return_sequences=True),
            tf.keras.layers.LSTM(32),
            tf.keras.layers.Dense(16, activation="relu"),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(1, activation="sigmoid"),
        ]
    )

    model_stacked_lstm.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

    print("堆叠LSTM模型:")
    model_stacked_lstm.summary()

    # 训练堆叠LSTM模型
    print("\n训练堆叠LSTM模型...")
    history_stacked = model_stacked_lstm.fit(
        x_train, y_train, epochs=10, batch_size=256, validation_data=(x_val, y_val), verbose=1
    )

    # 评估所有模型
    print("\nRNN模型评估:")
    models = [
        ("简单RNN", model_rnn, history_rnn),
        ("LSTM", model_lstm, history_lstm),
        ("双向LSTM", model_bilstm, history_bilstm),
        ("堆叠LSTM", model_stacked_lstm, history_stacked),
    ]

    for name, model, history in models:
        test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
        print(f"{name}: 测试准确率 = {test_acc:.4f}")

    return models


# =============================
# 4. 使用1D卷积的文本分类
# =============================


def text_classification_with_cnn(x_train, y_train, x_test, y_test, word_index):
    """
    使用1D卷积的文本分类
    """
    print("\n" + "=" * 50)
    print("使用1D卷积的文本分类")
    print("=" * 50)

    # 分割验证集
    x_val = x_train[:5000]
    y_val = y_train[:5000]
    x_train = x_train[5000:]
    y_train = y_train[5000:]

    vocab_size = len(word_index) + 1
    embedding_dim = 100
    max_length = 256

    # 1D CNN模型
    model_cnn = tf.keras.Sequential(
        [
            # 嵌入层
            tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
            # 多个不同大小的卷积核
            tf.keras.layers.Conv1D(128, 3, activation="relu"),
            tf.keras.layers.GlobalMaxPooling1D(),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(1, activation="sigmoid"),
        ]
    )

    model_cnn.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

    print("1D CNN模型:")
    model_cnn.summary()

    # 训练模型
    print("\n训练1D CNN模型...")
    history_cnn = model_cnn.fit(
        x_train, y_train, epochs=15, batch_size=256, validation_data=(x_val, y_val), verbose=1
    )

    # 更复杂的多尺度1D CNN模型
    print("\n多尺度1D CNN模型:")
    inputs = tf.keras.Input(shape=(max_length,))
    x = tf.keras.layers.Embedding(vocab_size, embedding_dim)(inputs)

    # 不同大小的卷积核
    conv3 = tf.keras.layers.Conv1D(128, 3, activation="relu")(x)
    conv3 = tf.keras.layers.GlobalMaxPooling1D()(conv3)

    conv5 = tf.keras.layers.Conv1D(128, 5, activation="relu")(x)
    conv5 = tf.keras.layers.GlobalMaxPooling1D()(conv5)

    conv7 = tf.keras.layers.Conv1D(128, 7, activation="relu")(x)
    conv7 = tf.keras.layers.GlobalMaxPooling1D()(conv7)

    # 合并所有卷积层的输出
    merged = tf.keras.layers.concatenate([conv3, conv5, conv7])

    # 分类层
    outputs = tf.keras.layers.Dense(1, activation="sigmoid")(merged)

    model_multi_cnn = tf.keras.Model(inputs, outputs)

    model_multi_cnn.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

    print("多尺度1D CNN模型:")
    model_multi_cnn.summary()

    # 训练多尺度CNN模型
    print("\n训练多尺度1D CNN模型...")
    history_multi_cnn = model_multi_cnn.fit(
        x_train, y_train, epochs=15, batch_size=256, validation_data=(x_val, y_val), verbose=1
    )

    # 评估模型
    print("\nCNN模型评估:")
    test_loss1, test_acc1 = model_cnn.evaluate(x_test, y_test, verbose=0)
    test_loss2, test_acc2 = model_multi_cnn.evaluate(x_test, y_test, verbose=0)

    print(f"单尺度CNN: 测试准确率 = {test_acc1:.4f}")
    print(f"多尺度CNN: 测试准确率 = {test_acc2:.4f}")

    return [("单尺度CNN", model_cnn, history_cnn), ("多尺度CNN", model_multi_cnn, history_multi_cnn)]


# =============================
# 5. 可视化和评估工具
# =============================


def plot_model_comparison(models, filename):
    """
    比较不同模型的性能
    """
    plt.figure(figsize=(12, 5))

    # 绘制准确率比较
    plt.subplot(1, 2, 1)
    for name, _, history in models:
        plt.plot(history.history["val_accuracy"], label=f"{name} (验证)")
    plt.title("模型准确率比较")
    plt.xlabel("Epoch")
    plt.ylabel("验证准确率")
    plt.legend()
    plt.grid(True)

    # 绘制损失比较
    plt.subplot(1, 2, 2)
    for name, _, history in models:
        plt.plot(history.history["val_loss"], label=f"{name} (验证)")
    plt.title("模型损失比较")
    plt.xlabel("Epoch")
    plt.ylabel("验证损失")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(filename)
    print(f"模型比较图表已保存为 '{filename}'")
    plt.close()


def analyze_predictions(model, x_test, y_test, word_index, num_samples=5):
    """
    分析模型预测结果
    """
    # 获取预测结果
    predictions = model.predict(x_test[:num_samples])
    predicted_probs = predictions.flatten()
    predicted_classes = (predictions > 0.5).astype(int).flatten()
    true_classes = y_test[:num_samples]

    # 反向词索引
    reverse_word_index = {v: k for k, v in word_index.items()}

    def decode_review(text):
        """将数字序列解码为文本"""
        return " ".join([reverse_word_index.get(i, "?") for i in text if i != 0])

    print("\n预测结果分析:")
    print("=" * 50)
    for i in range(num_samples):
        print(f"\n样本 {i+1}:")
        text = decode_review(x_test[i])
        print(f"文本: {text[:100]}...")  # 只显示前100个字符
        print(f"真实标签: {'正面' if true_classes[i] == 1 else '负面'}")
        print(f"预测概率: {predicted_probs[i]:.4f}")
        print(f"预测标签: {'正面' if predicted_classes[i] == 1 else '负面'}")
        print(f"预测正确: {'是' if predicted_classes[i] == true_classes[i] else '否'}")


# =============================
# 主函数
# =============================


def main():
    """
    主函数，执行所有文本分类示例
    """
    print("TensorFlow 2.0 文本分类实用案例")
    print("=" * 60)

    # 准备数据
    (x_train, y_train), (x_test, y_test), word_index = prepare_imdb_data()

    try:
        # 1. 使用词嵌入的分类
        print("\n1. 使用词嵌入的分类")
        embedding_models = text_classification_with_embeddings(
            x_train, y_train, x_test, y_test, word_index
        )

        # 2. 使用RNN的分类
        print("\n2. 使用RNN的分类")
        rnn_models = text_classification_with_rnn(x_train, y_train, x_test, y_test, word_index)

        # 3. 使用CNN的分类
        print("\n3. 使用CNN的分类")
        cnn_models = text_classification_with_cnn(x_train, y_train, x_test, y_test, word_index)

        # 4. 模型比较和可视化
        print("\n4. 模型性能比较")
        best_embedding_model = max(embedding_models, key=lambda x: x[2].history["val_accuracy"][-1])
        best_rnn_model = max(rnn_models, key=lambda x: x[2].history["val_accuracy"][-1])
        best_cnn_model = max(cnn_models, key=lambda x: x[2].history["val_accuracy"][-1])

        print("\n各类别最佳模型:")
        print(f"词嵌入最佳: {best_embedding_model[0]}")
        print(f"RNN最佳: {best_rnn_model[0]}")
        print(f"CNN最佳: {best_cnn_model[0]}")

        # 可视化比较
        plot_model_comparison(
            [best_embedding_model, best_rnn_model, best_cnn_model],
            "text_classification_comparison.png",
        )

        # 分析最佳模型的预测
        print("\n5. 最佳模型预测分析")
        overall_best_model = max(
            [best_embedding_model, best_rnn_model, best_cnn_model],
            key=lambda x: x[2].history["val_accuracy"][-1],
        )[1]

        analyze_predictions(overall_best_model, x_test, y_test, word_index)

    except Exception as e:
        print(f"执行过程中出现错误: {e}")
        print("这可能是由于缺少某些依赖或环境配置问题")

    print("\n" + "=" * 60)
    print("文本分类案例学习完成！")
    print("\n关键要点:")
    print("1. 词嵌入是文本分类的基础技术")
    print("2. RNN/LSTM能捕捉序列依赖关系")
    print("3. CNN能捕捉局部模式，速度快")
    print("4. 预训练嵌入可以提高性能")
    print("5. 不同模型适用于不同的文本分类任务")


if __name__ == "__main__":
    main()
