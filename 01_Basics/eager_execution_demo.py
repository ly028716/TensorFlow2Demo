"""
TensorFlow 2.0 即时执行(Eager Execution)演示

即时执行是TensorFlow 2.0的默认模式，它使TensorFlow操作
立即执行并返回具体的值，而不是构建计算图。
"""

import tensorflow as tf
import numpy as np
import time


def eager_execution_basics():
    """
    演示即时执行的基本特点
    """
    print("=" * 50)
    print("即时执行基础演示")
    print("=" * 50)

    # 即时执行状态下，操作立即返回结果
    print("1. 操作立即执行:")
    a = tf.constant([[1, 2], [3, 4]])
    b = tf.constant([[5, 6], [7, 8]])

    print(f"张量a:\n{a}")
    print(f"张量b:\n{b}")

    # 立即得到结果
    c = tf.matmul(a, b)
    print(f"矩阵乘法结果:\n{c}")
    print(f"结果类型: {type(c)}")

    # 可以直接访问值
    print(f"直接访问值: {c.numpy()}")

    # 支持Python控制流
    print("\n2. 支持Python控制流:")

    def dynamic_computation(x):
        if tf.reduce_sum(x) > 10:
            return x * 2
        else:
            return x / 2

    test_tensor1 = tf.constant([1, 2, 3])  # 和为6
    test_tensor2 = tf.constant([5, 5, 2])  # 和为12

    print(f"输入{test_tensor1.numpy()}, 输出{dynamic_computation(test_tensor1).numpy()}")
    print(f"输入{test_tensor2.numpy()}, 输出{dynamic_computation(test_tensor2).numpy()}")


def eager_vs_graph_mode():
    """
    比较即时执行模式和图模式的性能差异
    """
    print("\n" + "=" * 50)
    print("即时执行 vs 图模式性能比较")
    print("=" * 50)

    # 定义相同的函数，一个即时执行，一个图执行
    def compute_in_eager(x):
        """即时执行函数"""
        for i in range(100):
            x = tf.matmul(x, x)
        return x

    @tf.function
    def compute_in_graph(x):
        """图执行函数"""
        for i in range(100):
            x = tf.matmul(x, x)
        return x

    # 创建测试数据
    size = 100
    x = tf.random.normal((size, size))

    # 测试即时执行
    print("测试即时执行性能...")
    start_time = time.time()
    result_eager = compute_in_eager(x)
    eager_time = time.time() - start_time

    # 测试图执行
    print("测试图执行性能...")
    start_time = time.time()
    result_graph = compute_in_graph(x)
    graph_time = time.time() - start_time

    # 比较结果
    print(f"\n性能比较:")
    print(f"即时执行时间: {eager_time:.4f}秒")
    print(f"图执行时间: {graph_time:.4f}秒")
    print(f"性能提升: {eager_time/graph_time:.2f}倍")

    # 验证结果是否相同
    print(f"结果是否相同: {tf.reduce_all(tf.abs(result_eager - result_graph) < 1e-6)}")


def debugging_with_eager():
    """
    演示即时执行如何帮助调试
    """
    print("\n" + "=" * 50)
    print("即时执行与调试")
    print("=" * 50)

    def compute_with_potential_error(x):
        """一个可能有错误的计算函数"""
        print(f"输入张量形状: {x.shape}")
        print(f"输入张量值: {x}")

        # 故意引入一个错误 - 维度不匹配
        wrong_vector = tf.constant([1, 2, 3])  # 只有3个元素

        try:
            # 这会导致错误，因为x有更多列
            result = tf.matmul(x, tf.reshape(wrong_vector, (3, 1)))
            print(f"计算结果: {result}")
        except tf.errors.InvalidArgumentError as e:
            print(f"捕获到TensorFlow错误: {e}")
            # 在即时执行模式下，我们可以容易地修复这个问题
            print("在即时执行模式下可以立即修复并继续...")

            # 修复错误
            correct_vector = tf.constant([1, 2, 3, 4, 5][: x.shape[1]])  # 截取合适长度
            result = tf.matmul(x, tf.reshape(correct_vector, (x.shape[1], 1)))
            print(f"修复后的计算结果: {result}")

        return result

    # 测试调试函数
    test_matrix = tf.constant([[1, 2, 3, 4], [5, 6, 7, 8]])
    compute_with_potential_error(test_matrix)


def eager_with_numpy():
    """
    演示即时执行与NumPy的无缝集成
    """
    print("\n" + "=" * 50)
    print("即时执行与NumPy集成")
    print("=" * 50)

    # NumPy数组与TensorFlow张量的无缝转换
    print("1. NumPy数组与TensorFlow张量转换:")
    np_array = np.random.randn(3, 4)
    tf_tensor = tf.constant(np_array)

    print(f"NumPy数组:\n{np_array}")
    print(f"类型: {type(np_array)}, 形状: {np_array.shape}")
    print(f"转换为TensorFlow张量:")
    print(f"类型: {type(tf_tensor)}, 形状: {tf_tensor.shape}")

    # TensorFlow操作后的结果可以轻松转换为NumPy
    result = tf.reduce_sum(tf.square(tf_tensor), axis=0)
    print(f"TensorFlow操作结果: {result}")
    print(f"转换为NumPy数组: {result.numpy()}")
    print(f"转换后类型: {type(result.numpy())}")

    # 混合使用NumPy和TensorFlow函数
    print("\n2. 混合使用NumPy和TensorFlow:")

    def hybrid_computation(x):
        """混合使用NumPy和TensorFlow的函数"""
        # TensorFlow操作
        tf_result = tf.reduce_mean(x)

        # 转换为NumPy进行其他计算
        np_result = np.sqrt(tf_result.numpy())

        # 再次转换为TensorFlow
        final_result = tf.constant(np_result)

        return final_result

    test_tensor = tf.constant([1.0, 4.0, 9.0, 16.0])
    result = hybrid_computation(test_tensor)
    print(f"输入: {test_tensor}")
    print(f"混合计算结果: {result}")


def eager_and_keras():
    """
    演示即时执行与Keras的集成
    """
    print("\n" + "=" * 50)
    print("即时执行与Keras集成")
    print("=" * 50)

    # 创建简单的神经网络
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Dense(10, activation="relu", input_shape=(4,)),
            tf.keras.layers.Dense(3, activation="softmax"),
        ]
    )

    # 在即时执行模式下可以直接使用模型
    print("1. 模型前向传播:")
    sample_input = tf.random.normal((1, 4))
    output = model(sample_input)
    print(f"输入: {sample_input}")
    print(f"输出: {output}")
    print(f"输出形状: {output.shape}")

    # 查看模型参数
    print("\n2. 查看模型参数:")
    for i, layer in enumerate(model.layers):
        weights, biases = layer.get_weights()
        print(f"第{i+1}层 - 权重形状: {weights.shape}, 偏置形状: {biases.shape}")

    # 即时计算梯度
    print("\n3. 即时计算梯度:")
    with tf.GradientTape() as tape:
        predictions = model(sample_input)
        loss = tf.reduce_mean(predictions**2)  # 简单的损失函数

    # 计算梯度
    gradients = tape.gradient(loss, model.trainable_variables)

    print(f"损失值: {loss.numpy()}")
    print("可训练变量的梯度:")
    for i, (var, grad) in enumerate(zip(model.trainable_variables, gradients)):
        print(f"  变量{i+1} {var.name}: 梯度形状 = {grad.shape if grad is not None else 'None'}")


def main():
    """
    主函数
    """
    print("TensorFlow 2.0 即时执行演示")
    print("=" * 60)

    eager_execution_basics()
    eager_vs_graph_mode()
    debugging_with_eager()
    eager_with_numpy()
    eager_and_keras()

    print("\n" + "=" * 60)
    print("即时执行演示完成！")
    print("\n即时执行的主要优势:")
    print("1. 更直观的编程模型")
    print("2. 更容易调试")
    print("3. 与Python和NumPy无缝集成")
    print("4. 支持标准的Python控制流")
    print("5. 减少样板代码")


if __name__ == "__main__":
    main()
