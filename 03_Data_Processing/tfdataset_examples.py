"""
TensorFlow 2.0 tf.data 实用示例

本模块展示tf.data API的实际应用场景
包括文本数据、图像数据、时间序列数据的处理
"""

import tensorflow as tf
import numpy as np
import os
import pandas as pd


# =============================
# 1. 文本数据处理流水线
# =============================


class TextDataPipeline:
    """
    文本数据处理流水线
    """

    def __init__(self, vocab_size=10000, max_length=50):
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.tokenizer = None
        self.word_index = None

    def build_vocab(self, text_list):
        """
        构建词汇表
        """
        # 使用Keras的Tokenizer
        self.tokenizer = tf.keras.preprocessing.text.Tokenizer(
            num_words=self.vocab_size, oov_token="<UNK>"
        )
        self.tokenizer.fit_on_texts(text_list)
        self.word_index = self.tokenizer.word_index
        print(f"词汇表大小: {len(self.word_index)}")

    def text_to_sequence(self, text):
        """
        将文本转换为序列
        """
        if not self.tokenizer:
            raise ValueError("请先调用build_vocab构建词汇表")
        sequence = self.tokenizer.texts_to_sequences([text])[0]
        return sequence

    def pad_sequence(self, sequence):
        """
        填充或截断序列到固定长度
        """
        if len(sequence) > self.max_length:
            return sequence[: self.max_length]
        else:
            return sequence + [0] * (self.max_length - len(sequence))

    def create_dataset(self, texts, labels, batch_size=32, shuffle=True):
        """
        创建文本数据集
        """
        if not self.tokenizer:
            raise ValueError("请先调用build_vocab构建词汇表")

        # 将文本转换为序列
        sequences = self.tokenizer.texts_to_sequences(texts)
        padded_sequences = tf.keras.preprocessing.sequence.pad_sequences(
            sequences, maxlen=self.max_length, padding="post", truncating="post"
        )

        # 创建数据集
        dataset = tf.data.Dataset.from_tensor_slices((padded_sequences, labels))

        # 打乱数据
        if shuffle:
            dataset = dataset.shuffle(buffer_size=len(texts))

        # 批处理
        dataset = dataset.batch(batch_size)

        # 预取
        dataset = dataset.prefetch(tf.data.AUTOTUNE)

        return dataset

    def create_generator_dataset(self, text_files, labels, batch_size=32):
        """
        从文件生成器创建数据集
        """

        def generator():
            for text_file, label in zip(text_files, labels):
                with tf.io.gfile.GFile(text_file, "r") as f:
                    text = f.read().strip()
                sequence = self.text_to_sequence(text)
                sequence = self.pad_sequence(sequence)
                yield sequence, label

        output_signature = (
            tf.TensorSpec(shape=(self.max_length,), dtype=tf.int32),
            tf.TensorSpec(shape=(), dtype=tf.int32),
        )

        dataset = tf.data.Dataset.from_generator(generator, output_signature=output_signature)

        # 打乱、批处理和预取
        dataset = dataset.shuffle(buffer_size=len(text_files))
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)

        return dataset


# =============================
# 2. 图像数据处理流水线
# =============================


class ImageDataPipeline:
    """
    图像数据处理流水线
    """

    def __init__(self, image_size=(224, 224), batch_size=32):
        self.image_size = image_size
        self.batch_size = batch_size

    def parse_image(self, filename, label):
        """
        解析单个图像文件
        """
        # 读取图像文件
        image = tf.io.read_file(filename)
        # 解码图像 (假设是JPEG格式)
        image = tf.image.decode_jpeg(image, channels=3)
        # 转换为float32
        image = tf.image.convert_image_dtype(image, tf.float32)
        # 调整大小
        image = tf.image.resize(image, self.image_size)
        return image, label

    def augment_image(self, image, label):
        """
        图像增强
        """
        # 随机水平翻转
        image = tf.image.random_flip_left_right(image)

        # 随机调整亮度
        image = tf.image.random_brightness(image, max_delta=0.2)

        # 随机调整对比度
        image = tf.image.random_contrast(image, lower=0.8, upper=1.2)

        # 随机调整饱和度
        image = tf.image.random_saturation(image, lower=0.8, upper=1.2)

        # 确保像素值在[0,1]范围内
        image = tf.clip_by_value(image, 0.0, 1.0)

        return image, label

    def create_dataset(self, image_files, labels, augment=False, shuffle=True):
        """
        创建图像数据集
        """
        # 创建数据集
        dataset = tf.data.Dataset.from_tensor_slices((image_files, labels))

        # 打乱数据
        if shuffle:
            dataset = dataset.shuffle(buffer_size=len(image_files))

        # 并行解析图像
        dataset = dataset.map(self.parse_image, num_parallel_calls=tf.data.AUTOTUNE)

        # 数据增强 (仅用于训练集)
        if augment:
            dataset = dataset.map(self.augment_image, num_parallel_calls=tf.data.AUTOTUNE)

        # 批处理
        dataset = dataset.batch(self.batch_size)

        # 预取
        dataset = dataset.prefetch(tf.data.AUTOTUNE)

        return dataset

    def create_tfrecord_dataset(self, tfrecord_files, augment=False):
        """
        从TFRecord文件创建数据集
        """
        # TFRecord特征解析函数
        feature_description = {
            "image": tf.io.FixedLenFeature([], tf.string),
            "label": tf.io.FixedLenFeature([], tf.int64),
        }

        def _parse_function(example_proto):
            """解析单个TFRecord样本"""
            parsed_features = tf.io.parse_single_example(example_proto, feature_description)
            image = tf.io.decode_jpeg(parsed_features["image"], channels=3)
            image = tf.image.convert_image_dtype(image, tf.float32)
            image = tf.image.resize(image, self.image_size)
            label = tf.cast(parsed_features["label"], tf.int32)
            return image, label

        # 创建数据集
        dataset = tf.data.TFRecordDataset(tfrecord_files)
        dataset = dataset.map(_parse_function, num_parallel_calls=tf.data.AUTOTUNE)

        # 数据增强
        if augment:
            dataset = dataset.map(self.augment_image, num_parallel_calls=tf.data.AUTOTUNE)

        # 批处理和预取
        dataset = dataset.batch(self.batch_size)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)

        return dataset


# =============================
# 3. 时间序列数据处理流水线
# =============================


class TimeSeriesDataPipeline:
    """
    时间序列数据处理流水线
    """

    def __init__(self, window_size=10, batch_size=32):
        self.window_size = window_size
        self.batch_size = batch_size

    def sliding_window_dataset(self, data, targets, shift=1, drop_remainder=True):
        """
        创建滑动窗口数据集
        """
        # 创建窗口数据集
        dataset = tf.data.Dataset.from_tensor_slices(data)
        dataset = dataset.window(self.window_size, shift=shift, drop_remainder=drop_remainder)

        # 将窗口展平为批次
        dataset = dataset.flat_map(lambda window: window.batch(self.window_size))

        # 如果有目标值，创建对应的目标数据集
        if targets is not None:
            targets_dataset = tf.data.Dataset.from_tensor_slices(targets)
            # 调整目标数据集大小
            if drop_remainder:
                targets_dataset = targets_dataset.skip(self.window_size - 1)
            else:
                targets_dataset = targets_dataset

            # 组合特征和目标
            dataset = tf.data.Dataset.zip((dataset, targets_dataset))

        return dataset

    def create_forecasting_dataset(self, data, forecast_horizon=1, shift=1):
        """
        创建时间序列预测数据集
        """
        # 输入序列
        input_dataset = tf.data.Dataset.from_tensor_slices(data)
        input_dataset = input_dataset.window(self.window_size, shift=shift, drop_remainder=True)
        input_dataset = input_dataset.flat_map(lambda window: window.batch(self.window_size))

        # 目标序列 (预测未来forecast_horizon步)
        target_dataset = tf.data.Dataset.from_tensor_slices(data)
        target_dataset = target_dataset.skip(self.window_size)
        if forecast_horizon > 1:
            target_dataset = target_dataset.window(
                forecast_horizon, shift=shift, drop_remainder=True
            )
            target_dataset = target_dataset.flat_map(lambda window: window.batch(forecast_horizon))

        # 组合输入和目标
        dataset = tf.data.Dataset.zip((input_dataset, target_dataset))

        return dataset

    def create_dataset(self, data, targets=None, shuffle=True):
        """
        创建完整的时间序列数据集
        """
        if targets is not None:
            # 有监督学习
            dataset = self.sliding_window_dataset(data, targets)
        else:
            # 无监督学习或预测
            dataset = self.create_forecasting_dataset(data)

        # 打乱数据
        if shuffle:
            dataset = dataset.shuffle(buffer_size=1000)

        # 批处理
        dataset = dataset.batch(self.batch_size)

        # 预取
        dataset = dataset.prefetch(tf.data.AUTOTUNE)

        return dataset

    def normalize_data(self, data, method="z_score"):
        """
        标准化时间序列数据
        """
        if method == "z_score":
            mean = tf.reduce_mean(data)
            std = tf.math.reduce_std(data)
            normalized = (data - mean) / std
            return normalized, mean, std
        elif method == "min_max":
            min_val = tf.reduce_min(data)
            max_val = tf.reduce_max(data)
            normalized = (data - min_val) / (max_val - min_val)
            return normalized, min_val, max_val
        else:
            raise ValueError(f"不支持的标准化方法: {method}")


# =============================
# 4. 多模态数据处理
# =============================


class MultiModalDataPipeline:
    """
    多模态数据处理流水线
    """

    def __init__(self, image_size=(224, 224), max_length=50):
        self.image_size = image_size
        self.max_length = max_length
        self.text_processor = TextDataPipeline(max_length=max_length)

    def parse_sample(self, image_path, text, label):
        """
        解析单个多模态样本
        """
        # 处理图像
        image = tf.io.read_file(image_path)
        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.image.convert_image_dtype(image, tf.float32)
        image = tf.image.resize(image, self.image_size)

        # 处理文本
        if self.text_processor.tokenizer:
            sequence = self.text_processor.text_to_sequence(text.decode("utf-8"))
            sequence = self.text_processor.pad_sequence(sequence)
        else:
            sequence = tf.zeros(self.max_length, dtype=tf.int32)

        return (image, sequence), label

    def create_dataset(self, image_paths, texts, labels, batch_size=32, shuffle=True):
        """
        创建多模态数据集
        """
        # 创建数据集
        dataset = tf.data.Dataset.from_tensor_slices((image_paths, texts, labels))

        # 打乱数据
        if shuffle:
            dataset = dataset.shuffle(buffer_size=len(image_paths))

        # 解析样本
        dataset = dataset.map(self.parse_sample, num_parallel_calls=tf.data.AUTOTUNE)

        # 批处理
        dataset = dataset.batch(batch_size)

        # 预取
        dataset = dataset.prefetch(tf.data.AUTOTUNE)

        return dataset


# =============================
# 5. 示例和演示
# =============================


def text_pipeline_demo():
    """
    文本处理流水线演示
    """
    print("=" * 50)
    print("文本处理流水线演示")
    print("=" * 50)

    # 示例文本数据
    texts = [
        "I love deep learning",
        "TensorFlow is awesome",
        "Machine learning is fun",
        "Neural networks are powerful",
        "AI will change the world",
        "Data science is important",
        "Python is great for ML",
        "Deep learning needs big data",
    ]
    labels = [1, 1, 1, 1, 1, 0, 0, 0]  # 1: AI相关, 0: 其他

    # 创建文本处理流水线
    text_pipeline = TextDataPipeline(vocab_size=50, max_length=10)

    # 构建词汇表
    text_pipeline.build_vocab(texts)
    print(f"词汇表示例: {dict(list(text_pipeline.word_index.items())[:10])}")

    # 创建数据集
    dataset = text_pipeline.create_dataset(texts, labels, batch_size=4)

    print("\n文本数据集:")
    for batch in dataset.take(2):
        features, labels_batch = batch
        print(f"批次特征形状: {features.shape}")
        print(f"批次标签: {labels_batch.numpy()}")


def image_pipeline_demo():
    """
    图像处理流水线演示
    """
    print("\n" + "=" * 50)
    print("图像处理流水线演示")
    print("=" * 50)

    # 模拟图像文件路径和标签
    image_files = [f"image_{i}.jpg" for i in range(8)]
    labels = [0, 1, 0, 1, 0, 1, 0, 1]

    # 模拟图像解析函数 (因为实际上没有图像文件)
    def mock_parse_image(filename, label):
        """模拟图像解析"""
        image = tf.random.uniform((256, 256, 3))
        image = tf.image.resize(image, (224, 224))
        return image, label

    # 创建图像处理流水线
    image_pipeline = ImageDataPipeline(image_size=(224, 224), batch_size=4)

    # 创建模拟数据集
    dataset = tf.data.Dataset.from_tensor_slices((image_files, labels))
    dataset = dataset.map(mock_parse_image)

    # 模拟数据增强
    def mock_augment_image(image, label):
        """模拟图像增强"""
        # 随机翻转
        if tf.random.uniform([]) > 0.5:
            image = tf.image.flip_left_right(image)
        return image, label

    # 应用增强
    augmented_dataset = dataset.map(mock_augment_image)

    # 批处理
    batched_dataset = augmented_dataset.batch(4)

    print("图像数据集:")
    for batch in batched_dataset.take(2):
        images, labels_batch = batch
        print(f"批次图像形状: {images.shape}")
        print(f"批次标签: {labels_batch.numpy()}")


def time_series_pipeline_demo():
    """
    时间序列处理流水线演示
    """
    print("\n" + "=" * 50)
    print("时间序列处理流水线演示")
    print("=" * 50)

    # 创建模拟时间序列数据
    np.random.seed(42)
    time_series_data = np.cumsum(np.random.randn(100)) + 10
    targets = (time_series_data[5:] > 10).astype(int)  # 简单的目标值

    # 创建时间序列处理流水线
    ts_pipeline = TimeSeriesDataPipeline(window_size=5, batch_size=4)

    # 标准化数据
    normalized_data, mean, std = ts_pipeline.normalize_data(time_series_data[:95])
    print(f"原始数据范围: [{time_series_data.min():.2f}, {time_series_data.max():.2f}]")
    print(
        f"标准化后范围: [{float(tf.reduce_min(normalized_data)):.2f}, {float(tf.reduce_max(normalized_data)):.2f}]"
    )

    # 创建数据集
    dataset = ts_pipeline.create_dataset(normalized_data, targets)

    print("\n时间序列数据集:")
    for batch in dataset.take(2):
        inputs, targets_batch = batch
        print(f"批次输入形状: {inputs.shape}")
        print(f"批次目标: {targets_batch.numpy()}")


def performance_optimization_demo():
    """
    性能优化演示
    """
    print("\n" + "=" * 50)
    print("性能优化演示")
    print("=" * 50)

    # 创建大型数据集
    data_size = 10000
    data = tf.random.normal((data_size, 100))
    labels = tf.random.uniform((data_size,), maxval=10, dtype=tf.int32)

    # 基础数据集
    basic_dataset = tf.data.Dataset.from_tensor_slices((data, labels))

    # 优化方案1: 预取
    prefetch_dataset = basic_dataset.batch(32).prefetch(tf.data.AUTOTUNE)

    # 优化方案2: 缓存
    cache_dataset = basic_dataset.map(lambda x, y: (x, y)).cache().batch(32)

    # 优化方案3: 并行处理
    parallel_dataset = basic_dataset.map(
        lambda x, y: (x * 2, y * 2), num_parallel_calls=tf.data.AUTOTUNE
    ).batch(32)

    print("性能优化对比:")
    datasets = {
        "基础": basic_dataset.batch(32),
        "预取": prefetch_dataset,
        "缓存": cache_dataset,
        "并行": parallel_dataset,
    }

    for name, dataset in datasets.items():
        start_time = tf.timestamp()
        count = 0
        for _ in dataset.take(100):  # 只取100个批次进行测试
            count += 1
        elapsed = tf.timestamp() - start_time
        print(f"{name}处理100批次耗时: {float(elapsed):.4f}秒")


# =============================
# 主函数
# =============================


def main():
    """
    主函数，执行所有tf.data示例
    """
    print("TensorFlow 2.0 tf.data 实用示例")
    print("=" * 60)

    # 执行各个模块
    text_pipeline_demo()
    image_pipeline_demo()
    time_series_pipeline_demo()
    performance_optimization_demo()

    print("\n" + "=" * 60)
    print("tf.data实用示例完成！")
    print("\n关键要点:")
    print("1. tf.data提供高效的数据处理管道")
    print("2. 并行处理、预取和缓存能显著提高性能")
    print("3. 不同的数据类型需要专门的处理流水线")
    print("4. 数据增强可以提高模型泛化能力")
    print("5. 模块化设计使数据处理代码更易维护")


if __name__ == "__main__":
    main()
