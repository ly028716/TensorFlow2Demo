"""
TensorFlow 2.0 数据处理基础

本模块介绍TensorFlow 2.0中的数据处理功能
包括tf.data API、数据预处理、数据增强等
"""

import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os


# =============================
# 1. tf.data API 基础
# =============================


def tfdata_basics():
    """
    tf.data API 基础功能演示
    """
    print("=" * 50)
    print("tf.data API 基础")
    print("=" * 50)

    # 1. 从NumPy数组创建数据集
    print("1. 从NumPy数组创建数据集:")
    numpy_data = np.arange(10)
    dataset_from_array = tf.data.Dataset.from_tensor_slices(numpy_data)
    print(f"原始数组: {numpy_data}")
    print("从数组创建的Dataset:")
    for item in dataset_from_array.take(5):
        print(f"  {item.numpy()}")

    # 2. 从张量创建数据集
    print("\n2. 从张量创建数据集:")
    tensor_data = tf.range(5, 15)
    dataset_from_tensor = tf.data.Dataset.from_tensor_slices(tensor_data)
    print("从张量创建的Dataset:")
    for item in dataset_from_tensor.take(5):
        print(f"  {item.numpy()}")

    # 3. 创建特征和标签数据集
    print("\n3. 创建特征和标签数据集:")
    features = tf.random.normal((100, 4))
    labels = tf.random.uniform((100,), maxval=2, dtype=tf.int32)
    dataset = tf.data.Dataset.from_tensor_slices((features, labels))
    print("特征-标签对数据集:")
    for feature, label in dataset.take(3):
        print(f"  特征: {feature.numpy()}, 标签: {label.numpy()}")

    # 4. Dataset基本操作
    print("\n4. Dataset基本操作:")
    # take: 取前n个元素
    # skip: 跳过n个元素
    # map: 应用转换函数
    # filter: 过滤元素
    # batch: 批处理
    # repeat: 重复数据集

    # 创建一个简单的数据集
    simple_dataset = tf.data.Dataset.range(10)

    print(f"原始数据集: {[item.numpy() for item in simple_dataset]}")

    # take操作
    taken_dataset = simple_dataset.take(3)
    print(f"take(3): {[item.numpy() for item in taken_dataset]}")

    # skip操作
    skipped_dataset = simple_dataset.skip(5)
    print(f"skip(5): {[item.numpy() for item in skipped_dataset]}")

    # map操作
    mapped_dataset = simple_dataset.map(lambda x: x * 2)
    print(f"map(x*2): {[item.numpy() for item in mapped_dataset]}")

    # filter操作
    filtered_dataset = simple_dataset.filter(lambda x: x % 2 == 0)
    print(f"filter(x%2==0): {[item.numpy() for item in filtered_dataset]}")


# =============================
# 2. 数据预处理
# =============================


def data_preprocessing():
    """
    数据预处理技术演示
    """
    print("\n" + "=" * 50)
    print("数据预处理")
    print("=" * 50)

    # 1. 数值特征标准化
    print("1. 数值特征标准化:")
    # 创建模拟数据
    data = tf.random.normal((100, 3), mean=10, stddev=5)
    print(f"原始数据前3行:\n{data[:3]}")
    print(
        f"原始数据统计 - 均值: {tf.reduce_mean(data).numpy():.2f}, 标准差: {tf.math.reduce_std(data).numpy():.2f}"
    )

    # 标准化 (Z-score标准化)
    mean = tf.reduce_mean(data, axis=0)
    std = tf.math.reduce_std(data, axis=0)
    normalized_data = (data - mean) / std
    print(f"标准化后数据前3行:\n{normalized_data[:3]}")
    print(
        f"标准化后统计 - 均值: {tf.reduce_mean(normalized_data).numpy():.6f}, 标准差: {tf.math.reduce_std(normalized_data).numpy():.6f}"
    )

    # 2. 归一化 (Min-Max缩放)
    print("\n2. 归一化 (Min-Max缩放):")
    min_val = tf.reduce_min(data, axis=0)
    max_val = tf.reduce_max(data, axis=0)
    scaled_data = (data - min_val) / (max_val - min_val)
    print(f"归一化后数据前3行:\n{scaled_data[:3]}")
    print(
        f"归一化后统计 - 最小值: {tf.reduce_min(scaled_data).numpy():.6f}, 最大值: {tf.reduce_max(scaled_data).numpy():.6f}"
    )

    # 3. 类别特征编码
    print("\n3. 类别特征编码:")
    # 模拟类别特征
    categories = tf.constant(["red", "blue", "green", "blue", "red", "green"])
    print(f"原始类别: {categories.numpy()}")

    # 获取所有唯一类别
    unique_categories = tf.unique(categories)[0]
    print(f"唯一类别: {unique_categories.numpy()}")

    # 转换为整数索引
    _, indices = tf.unique(categories)
    print(f"转换为索引: {indices.numpy()}")

    # One-hot编码
    one_hot = tf.one_hot(indices, depth=len(unique_categories))
    print(f"One-hot编码:\n{one_hot.numpy()}")

    # 4. 文本数据处理
    print("\n4. 文本数据处理:")
    text_data = tf.constant(
        ["TensorFlow is great", "I love machine learning", "Deep learning is fun"]
    )
    print(f"原始文本: {text_data.numpy()}")

    # 创建词汇表
    tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=20)
    tokenizer.fit_on_texts([text.numpy().decode("utf-8") for text in text_data])
    word_index = tokenizer.word_index
    print(f"词汇表: {word_index}")

    # 转换为序列
    sequences = tokenizer.texts_to_sequences([text.numpy().decode("utf-8") for text in text_data])
    print(f"转换为序列: {sequences}")

    # 填充序列
    padded_sequences = tf.keras.preprocessing.sequence.pad_sequences(
        sequences, maxlen=5, padding="post", truncating="post"
    )
    print(f"填充后序列:\n{padded_sequences}")


# =============================
# 3. tf.data 高级操作
# =============================


def advanced_tfdata_operations():
    """
    tf.data API 高级操作演示
    """
    print("\n" + "=" * 50)
    print("tf.data API 高级操作")
    print("=" * 50)

    # 创建一个示例数据集
    dataset = tf.data.Dataset.range(20)

    # 1. 批处理和重复
    print("1. 批处理和重复:")
    batched_dataset = dataset.batch(4)
    print("batch(4):")
    for batch in batched_dataset.take(3):
        print(f"  批次: {batch.numpy()}")

    repeated_dataset = dataset.repeat(2)
    print(f"repeat(2)后的长度: {len(list(repeated_dataset.as_numpy_iterator()))}")

    batch_and_repeat = dataset.batch(4).repeat(2)
    print("batch(4).repeat(2):")
    for batch in batch_and_repeat.take(6):
        print(f"  批次: {batch.numpy()}")

    # 2. 窗口操作 (用于时间序列数据)
    print("\n2. 窗口操作:")
    windowed_dataset = dataset.window(3, shift=1, drop_remainder=True)
    print("window(3, shift=1):")
    for i, window in enumerate(windowed_dataset.take(3)):
        window_array = [item.numpy() for item in window]
        print(f"  窗口{i}: {window_array}")

    # 3. 缓存和预取
    print("\n3. 缓存和预取:")

    # 模拟一个计算密集型操作
    def expensive_operation(x):
        return tf.square(x)

    # 创建一个处理流水线
    processed_dataset = (
        dataset.map(expensive_operation)  # 计算密集型操作
        .cache()  # 缓存结果
        .batch(4)  # 批处理
        .prefetch(tf.data.AUTOTUNE)
    )  # 预取

    print("缓存和预取流水线:")
    for batch in processed_dataset.take(3):
        print(f"  处理后的批次: {batch.numpy()}")

    # 4. 并行处理
    print("\n4. 并行处理:")

    # 模拟一个IO密集型操作
    def io_intensive_operation(x):
        tf.py_function(lambda: None, [], [])  # 模拟IO等待
        return x * 2

    # 并行map处理
    parallel_dataset = dataset.map(
        io_intensive_operation, num_parallel_calls=tf.data.AUTOTUNE
    ).batch(4)

    print("并行处理流水线:")
    for batch in parallel_dataset.take(3):
        print(f"  并行处理的批次: {batch.numpy()}")

    # 5. 打乱数据
    print("\n5. 打乱数据:")
    shuffled_dataset = dataset.shuffle(buffer_size=10, seed=42)
    print("shuffle(buffer_size=10):")
    for item in shuffled_dataset.take(10):
        print(f"  {item.numpy()}", end=" ")
    print()


# =============================
# 4. 图像数据处理
# =============================


def image_data_processing():
    """
    图像数据处理演示
    """
    print("\n" + "=" * 50)
    print("图像数据处理")
    print("=" * 50)

    # 1. 读取图像文件
    print("1. 读取图像文件:")
    # 创建一个虚拟图像文件 (实际使用时替换为真实图像路径)
    dummy_image_path = "dummy_image.jpg"

    # 模拟图像读取函数
    def read_image(image_path):
        # 这里只是模拟，实际使用 tf.io.read_file 和 tf.image.decode_image
        dummy_image = tf.random.uniform((256, 256, 3), maxval=256, dtype=tf.int32)
        return tf.cast(dummy_image, tf.float32)

    image = read_image(dummy_image_path)
    print(f"模拟图像形状: {image.shape}")
    print(f"图像数据类型: {image.dtype}")

    # 2. 图像预处理
    print("\n2. 图像预处理:")
    # 调整大小
    resized_image = tf.image.resize(image, [224, 224])
    print(f"调整大小后: {resized_image.shape}")

    # 归一化到[0,1]
    normalized_image = image / 255.0
    print(f"归一化后范围: [{tf.reduce_min(normalized_image):.4f}, {tf.reduce_max(normalized_image):.4f}]")

    # 3. 图像增强
    print("\n3. 图像增强:")
    # 随机翻转
    flipped_image = tf.image.random_flip_left_right(image)
    print(f"随机翻转: {'是' if tf.reduce_any(tf.not_equal(image, flipped_image)) else '否'}")

    # 随机旋转
    rotated_image = tf.image.rot90(
        image, k=tf.random.uniform([], minval=0, maxval=4, dtype=tf.int32)
    )
    print(f"旋转后的形状: {rotated_image.shape}")

    # 随机亮度调整
    brightness_adjusted = tf.image.random_brightness(image, max_delta=0.2)
    print(f"亮度调整前: {tf.reduce_mean(image):.2f}")
    print(f"亮度调整后: {tf.reduce_mean(brightness_adjusted):.2f}")

    # 随机对比度调整
    contrast_adjusted = tf.image.random_contrast(image, lower=0.8, upper=1.2)
    print(f"对比度调整前: {tf.math.reduce_std(image):.2f}")
    print(f"对比度调整后: {tf.math.reduce_std(contrast_adjusted):.2f}")

    # 4. 创建图像数据集流水线
    print("\n4. 创建图像数据集流水线:")

    def process_image(image_path):
        """图像处理函数"""
        image = read_image(image_path)
        image = tf.image.resize(image, [224, 224])
        image = image / 255.0
        image = tf.image.random_flip_left_right(image)
        return image

    # 模拟多个图像路径
    image_paths = [f"image_{i}.jpg" for i in range(10)]

    # 创建图像数据集
    image_dataset = (
        tf.data.Dataset.from_tensor_slices(image_paths)
        .map(process_image, num_parallel_calls=tf.data.AUTOTUNE)
        .batch(4)
        .prefetch(tf.data.AUTOTUNE)
    )

    print("图像处理流水线:")
    for batch in image_dataset.take(2):
        print(f"  批次形状: {batch.shape}")


# =============================
# 5. CSV数据处理
# =============================


def csv_data_processing():
    """
    CSV数据处理演示
    """
    print("\n" + "=" * 50)
    print("CSV数据处理")
    print("=" * 50)

    # 1. 创建示例CSV数据
    print("1. 创建示例CSV数据:")
    csv_data = [
        "feature1,feature2,feature3,label",
        "1.2,3.4,5.6,0",
        "2.3,4.5,6.7,1",
        "3.4,5.6,7.8,0",
        "4.5,6.7,8.9,1",
        "5.6,7.8,9.0,0",
    ]

    # 将数据写入临时文件 (在实际应用中，文件已存在)
    csv_filename = "sample_data.csv"
    with tf.io.gfile.GFile(csv_filename, "w") as f:
        f.write("\n".join(csv_data))

    print(f"创建CSV文件: {csv_filename}")
    print(f"CSV内容:")
    for line in csv_data:
        print(f"  {line}")

    # 2. 使用tf.data读取CSV文件
    print("\n2. 使用tf.data读取CSV文件:")

    def parse_csv_line(line):
        """解析CSV行"""
        # 定义默认值
        record_defaults = [[0.0], [0.0], [0.0], [0]]
        # 解析一行
        fields = tf.io.decode_csv(line, record_defaults)
        # 提取特征和标签
        features = tf.stack(fields[:-1])  # 除了最后一个值都是特征
        label = fields[-1]  # 最后一个值是标签
        return features, label

    # 读取CSV文件
    csv_dataset = (
        tf.data.TextLineDataset(csv_filename).skip(1).map(parse_csv_line).batch(2)  # 跳过标题行
    )

    print("CSV数据集:")
    for features_batch, label_batch in csv_dataset.take(2):
        print(f"  特征: {features_batch.numpy()}")
        print(f"  标签: {label_batch.numpy()}")

    # 3. 使用pandas预处理后转换为tf.data
    print("\n3. 使用pandas预处理后转换为tf.data:")
    # 读取CSV到pandas
    import io

    df = pd.read_csv(csv_filename)
    print("Pandas数据:")
    print(df.head())

    # 数据预处理
    features_df = df.drop("label", axis=1)
    labels_df = df["label"]

    # 标准化特征
    features = (features_df - features_df.mean()) / features_df.std()
    print("标准化后的特征:")
    print(features.head())

    # 转换为tf.data
    features_tensor = tf.constant(features.values, dtype=tf.float32)
    labels_tensor = tf.constant(labels_df.values, dtype=tf.int32)

    processed_dataset = tf.data.Dataset.from_tensor_slices((features_tensor, labels_tensor))
    processed_dataset = processed_dataset.batch(2)

    print("预处理后的数据集:")
    for features_batch, label_batch in processed_dataset.take(2):
        print(f"  特征: {features_batch.numpy()}")
        print(f"  标签: {label_batch.numpy()}")


# =============================
# 6. 数据集分割和生成器
# =============================


def dataset_splitting_and_generators():
    """
    数据集分割和生成器演示
    """
    print("\n" + "=" * 50)
    print("数据集分割和生成器")
    print("=" * 50)

    # 1. 数据集分割
    print("1. 数据集分割:")
    # 创建完整数据集
    all_data = tf.data.Dataset.range(100)
    dataset_size = len(list(all_data.as_numpy_iterator()))

    print(f"总数据集大小: {dataset_size}")

    # 分割为训练集、验证集和测试集
    train_size = int(0.7 * dataset_size)
    val_size = int(0.2 * dataset_size)

    train_dataset = all_data.take(train_size)
    remaining = all_data.skip(train_size)
    val_dataset = remaining.take(val_size)
    test_dataset = remaining.skip(val_size)

    print(f"训练集大小: {len(list(train_dataset.as_numpy_iterator()))}")
    print(f"验证集大小: {len(list(val_dataset.as_numpy_iterator()))}")
    print(f"测试集大小: {len(list(test_dataset.as_numpy_iterator()))}")

    # 2. 使用Python生成器创建数据集
    print("\n2. 使用Python生成器创建数据集:")

    def data_generator():
        """Python数据生成器"""
        for i in range(20):
            yield i, i * 2  # 产生(索引, 值)对

    # 从生成器创建数据集
    output_signature = (
        tf.TensorSpec(shape=(), dtype=tf.int32),
        tf.TensorSpec(shape=(), dtype=tf.int32),
    )

    generator_dataset = tf.data.Dataset.from_generator(
        data_generator, output_signature=output_signature
    )

    print("生成器数据集:")
    for item in generator_dataset.take(5):
        print(f"  索引: {item[0].numpy()}, 值: {item[1].numpy()}")

    # 3. 无限数据生成器
    print("\n3. 无限数据生成器:")

    def infinite_generator():
        """无限数据生成器"""
        i = 0
        while True:
            yield i, i % 3  # 产生(索引, 类别)对
            i += 1

    infinite_dataset = (
        tf.data.Dataset.from_generator(infinite_generator, output_signature=output_signature)
        .batch(3)
        .take(3)
    )  # 只取3个批次用于演示

    print("无限生成器数据集:")
    for batch in infinite_dataset:
        indices, values = batch
        print(f"  批次索引: {indices.numpy()}, 批次值: {values.numpy()}")


# =============================
# 主函数
# =============================


def main():
    """
    主函数，执行所有数据处理示例
    """
    print("TensorFlow 2.0 数据处理基础学习")
    print("=" * 60)

    # 执行各个模块
    tfdata_basics()
    data_preprocessing()
    advanced_tfdata_operations()
    image_data_processing()
    csv_data_processing()
    dataset_splitting_and_generators()

    print("\n" + "=" * 60)
    print("数据处理基础学习完成！")
    print("\n关键要点:")
    print("1. tf.data API提供高效的数据处理管道")
    print("2. 批处理、预取和缓存可以显著提高性能")
    print("3. 并行处理可以加速数据预处理")
    print("4. 数据增强可以提高模型泛化能力")
    print("5. 适当的数据预处理对模型性能至关重要")


if __name__ == "__main__":
    main()
