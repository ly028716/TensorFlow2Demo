"""
TensorFlow 2.0 基础语法知识讲解

本模块介绍TensorFlow 2.0的核心概念和基本操作
包括张量、变量、自动微分等基础内容
"""

import tensorflow as tf
import numpy as np


# =============================
# 1. TensorFlow 2.0 简介
# =============================


def tensorflow_introduction():
    """
    TensorFlow 2.0 简介和特点
    """
    print("=" * 50)
    print("TensorFlow 2.0 简介")
    print("=" * 50)

    # 打印TensorFlow版本
    print(f"TensorFlow版本: {tf.__version__}")

    # 检查GPU是否可用
    print(f"GPU是否可用: {tf.config.list_physical_devices('GPU')}")

    print("\nTensorFlow 2.0 主要特点:")
    print("- 默认启用即时执行(Eager Execution)")
    print("- 使用Keras作为高级API")
    print("- 更简洁的API设计")
    print("- 更好的Python集成")
    print("- 兼容TensorFlow 1.x的代码")


# =============================
# 2. 张量(Tensors)基础
# =============================


def tensor_basics():
    """
    张量基础知识讲解
    张量是TensorFlow中的基本数据结构
    """
    print("\n" + "=" * 50)
    print("张量(Tensors)基础")
    print("=" * 50)

    # 创建常量张量
    print("创建常量张量:")
    scalar = tf.constant(42)  # 0维张量(标量)
    vector = tf.constant([1, 2, 3])  # 1维张量(向量)
    matrix = tf.constant([[1, 2], [3, 4]])  # 2维张量(矩阵)
    tensor_3d = tf.constant([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])  # 3维张量

    print(f"标量: {scalar}, 维度: {scalar.ndim}, 形状: {scalar.shape}")
    print(f"向量: {vector}, 维度: {vector.ndim}, 形状: {vector.shape}")
    print(f"矩阵: \n{matrix}, 维度: {matrix.ndim}, 形状: {matrix.shape}")
    print(f"3维张量: \n{tensor_3d}, 维度: {tensor_3d.ndim}, 形状: {tensor_3d.shape}")

    # 创建特殊张量
    print("\n创建特殊张量:")
    zeros_tensor = tf.zeros(shape=(2, 3))
    ones_tensor = tf.ones(shape=(2, 3))
    random_tensor = tf.random.normal(shape=(2, 3))

    print(f"全零张量:\n{zeros_tensor}")
    print(f"全一张量:\n{ones_tensor}")
    print(f"随机正态分布张量:\n{random_tensor}")

    # 张量与NumPy数组的转换
    print("\n张量与NumPy数组的转换:")
    numpy_array = np.array([[1, 2, 3], [4, 5, 6]])
    tensor_from_numpy = tf.constant(numpy_array)
    numpy_from_tensor = tensor_from_numpy.numpy()

    print(f"NumPy数组: {type(numpy_array)}\n{numpy_array}")
    print(f"从NumPy创建的张量: {type(tensor_from_numpy)}\n{tensor_from_numpy}")
    print(f"转换为NumPy数组: {type(numpy_from_tensor)}\n{numpy_from_tensor}")


# =============================
# 3. 张量操作
# =============================


def tensor_operations():
    """
    张量的数学运算和操作
    """
    print("\n" + "=" * 50)
    print("张量操作")
    print("=" * 50)

    # 创建示例张量
    a = tf.constant([[1, 2], [3, 4]])
    b = tf.constant([[5, 6], [7, 8]])

    print(f"张量a:\n{a}")
    print(f"张量b:\n{b}")

    # 基本数学运算
    print("\n基本数学运算:")
    print(f"加法: a + b =\n{tf.add(a, b)}")
    print(f"减法: a - b =\n{tf.subtract(a, b)}")
    print(f"乘法(元素): a * b =\n{tf.multiply(a, b)}")
    print(f"除法: a / b =\n{tf.divide(a, b)}")

    # 矩阵乘法
    print(f"\n矩阵乘法: a @ b =\n{tf.matmul(a, b)}")

    # 其他数学运算
    print("\n其他数学运算:")
    x = tf.constant([1.0, 2.0, 3.0, 4.0])
    print(f"平方: tf.square({x}) = {tf.square(x)}")
    print(f"平方根: tf.sqrt({x}) = {tf.sqrt(x)}")
    print(f"幂运算: tf.pow({x}, 3) = {tf.pow(x, 3)}")
    print(f"指数运算: tf.exp({x}) = {tf.exp(x)}")

    # 张量统计操作
    print("\n张量统计操作:")
    tensor_stats = tf.constant([[1, 2, 3], [4, 5, 6]])
    print(f"统计张量:\n{tensor_stats}")
    print(f"总和: tf.reduce_sum(tensor_stats) = {tf.reduce_sum(tensor_stats)}")
    print(f"按行求和: tf.reduce_sum(tensor_stats, axis=1) = {tf.reduce_sum(tensor_stats, axis=1)}")
    print(f"按列求和: tf.reduce_sum(tensor_stats, axis=0) = {tf.reduce_sum(tensor_stats, axis=0)}")
    print(f"平均值: tf.reduce_mean(tensor_stats) = {tf.reduce_mean(tensor_stats)}")
    print(f"最大值: tf.reduce_max(tensor_stats) = {tf.reduce_max(tensor_stats)}")
    print(f"最小值: tf.reduce_min(tensor_stats) = {tf.reduce_min(tensor_stats)}")

    # 张量形状操作
    print("\n张量形状操作:")
    x = tf.constant([[1, 2, 3, 4], [5, 6, 7, 8]])
    print(f"原始张量:\n{x}, 形状: {x.shape}")

    reshaped = tf.reshape(x, (4, 2))
    print(f"重塑后(4,2):\n{reshaped}, 形状: {reshaped.shape}")

    transposed = tf.transpose(x)
    print(f"转置后:\n{transposed}, 形状: {transposed.shape}")

    squeezed = tf.squeeze(tf.constant([[[1, 2, 3]]]))
    print(f"降维后: {squeezed}, 形状: {squeezed.shape}")


# =============================
# 4. 变量和自动微分
# =============================


def variables_and_gradients():
    """
    TensorFlow变量和自动微分
    """
    print("\n" + "=" * 50)
    print("变量和自动微分")
    print("=" * 50)

    # 变量创建
    print("创建变量:")
    variable = tf.Variable([1.0, 2.0, 3.0, 4.0])
    print(f"变量: {variable}, 值: {variable.numpy()}")

    # 变量操作
    variable.assign([5.0, 6.0, 7.0, 8.0])  # 直接赋值
    print(f"赋值后: {variable.numpy()}")

    variable.assign_add([1.0, 1.0, 1.0, 1.0])  # 加法赋值
    print(f"加法赋值后: {variable.numpy()}")

    variable.assign_sub([2.0, 2.0, 2.0, 2.0])  # 减法赋值
    print(f"减法赋值后: {variable.numpy()}")

    # 自动微分示例
    print("\n自动微分示例:")

    # 使用GradientTape记录操作
    x = tf.Variable(3.0)

    with tf.GradientTape() as tape:
        y = x**2  # y = x^2

    # 计算梯度 dy/dx = 2x = 6
    dy_dx = tape.gradient(y, x)
    print(f"当x = {x.numpy()}, y = x^2 = {y.numpy()}")
    print(f"梯度 dy/dx = {dy_dx.numpy()}")

    # 多元函数梯度计算
    print("\n多元函数梯度计算:")
    w = tf.Variable(tf.random.normal((2, 2)))
    b = tf.Variable(tf.zeros(2, dtype=tf.float32))
    x = tf.constant([[1.0, 2.0]])

    with tf.GradientTape() as tape:
        y = tf.matmul(x, w) + b  # y = x*w + b

    # 计算y关于w和b的梯度
    dy_dw, dy_db = tape.gradient(y, [w, b])
    print(f"输入x:\n{x}")
    print(f"权重w:\n{w}")
    print(f"偏置b: {b}")
    print(f"输出y: {y}")
    print(f"梯度dy/dw:\n{dy_dw}")
    print(f"梯度dy/db: {dy_db}")


# =============================
# 5. 即时执行与函数装饰器
# =============================


def eager_vs_function():
    """
    即时执行与tf.function装饰器
    """
    print("\n" + "=" * 50)
    print("即时执行与tf.function装饰器")
    print("=" * 50)

    # 即时执行
    print("即时执行:")

    def simple_function(x):
        return tf.reduce_sum(x**2)

    x = tf.constant([1, 2, 3, 4])
    print(f"输入: {x}")
    print(f"即时执行结果: {simple_function(x)}")

    # 使用tf.function装饰器转换为图
    print("\n使用tf.function装饰器:")

    @tf.function
    def graph_function(x):
        return tf.reduce_sum(x**2)

    print(f"图执行结果: {graph_function(x)}")

    # 比较性能
    print("\n性能比较:")
    import time

    # 测试即时执行
    start_time = time.time()
    for _ in range(1000):
        simple_function(x)
    eager_time = time.time() - start_time

    # 测试图执行
    start_time = time.time()
    for _ in range(1000):
        graph_function(x)
    graph_time = time.time() - start_time

    print(f"即时执行时间: {eager_time:.4f}秒")
    print(f"图执行时间: {graph_time:.4f}秒")
    print(f"性能提升: {eager_time/graph_time:.2f}倍")


# =============================
# 主函数
# =============================


def main():
    """
    主函数，执行所有TensorFlow基础示例
    """
    print("TensorFlow 2.0 基础语法学习")
    print("=" * 60)

    # 执行各个模块
    tensorflow_introduction()
    tensor_basics()
    tensor_operations()
    variables_and_gradients()
    eager_vs_function()

    print("\n" + "=" * 60)
    print("基础语法学习完成！")


if __name__ == "__main__":
    main()
