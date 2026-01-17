"""
TensorFlow 2.0 学习项目演示脚本

这个脚本运行一个简短的演示，展示项目的主要功能
适合初学者快速了解TensorFlow 2.0的基本用法
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# 添加当前目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 导入工具函数（使用importlib因为模块名以数字开头）
import importlib.util
spec = importlib.util.spec_from_file_location(
    "learning_tools",
    os.path.join(os.path.dirname(__file__), "06_Utils", "learning_tools.py")
)
learning_tools = importlib.util.module_from_spec(spec)
spec.loader.exec_module(learning_tools)
plot_training_history = learning_tools.plot_training_history
count_parameters = learning_tools.count_parameters


def quick_demo():
    """
    快速演示TensorFlow 2.0的核心功能
    """
    print("="*60)
    print("TensorFlow 2.0 快速演示")
    print("="*60)

    # 1. 显示TensorFlow版本
    print(f"\n1. TensorFlow版本: {tf.__version__}")

    # 2. 检查GPU是否可用
    gpu_available = len(tf.config.list_physical_devices('GPU')) > 0
    print(f"   GPU可用: {'是' if gpu_available else '否'}")

    # 3. 基础张量操作
    print("\n2. 基础张量操作:")
    # 创建张量
    a = tf.constant([[1, 2], [3, 4]], dtype=tf.float32)
    b = tf.constant([[5, 6], [7, 8]], dtype=tf.float32)

    print(f"   张量A:\n{a}")
    print(f"   张量B:\n{b}")

    # 基本运算
    c = tf.add(a, b)
    d = tf.matmul(a, b)

    print(f"   A + B =\n{c}")
    print(f" A × B =\n{d}")

    # 4. 自动微分演示
    print("\n3. 自动微分演示:")
    x = tf.Variable(2.0)

    with tf.GradientTape() as tape:
        y = x**3 + 2*x**2 + x + 1

    dy_dx = tape.gradient(y, x)
    print(f"   当x = {x.numpy()}, y = {y.numpy()}")
    print(f"   dy/dx = {dy_dx.numpy()}")

    # 5. 简单神经网络演示
    print("\n4. 简单神经网络演示:")

    # 生成示例数据
    np.random.seed(42)
    X = np.random.randn(1000, 10).astype(np.float32)
    y = (np.sum(X, axis=1, keepdims=True) > 0).astype(np.float32)

    print(f"   数据形状: X={X.shape}, y={y.shape}")

    # 创建简单的神经网络
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(32, activation='relu', input_shape=(10,)),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    print(f"   模型架构:")
    model.summary()

    # 显示模型参数
    params_info = count_parameters(model)

    # 6. 模型编译和训练
    print("\n5. 模型训练:")
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    print("   开始训练...")
    history = model.fit(
        X, y,
        epochs=10,
        batch_size=32,
        validation_split=0.2,
        verbose=0  # 静默模式以减少输出
    )

    # 7. 训练结果
    print("\n6. 训练结果:")
    final_loss = history.history['loss'][-1]
    final_acc = history.history['accuracy'][-1]
    final_val_loss = history.history['val_loss'][-1]
    final_val_acc = history.history['val_accuracy'][-1]

    print(f"   最终训练损失: {final_loss:.4f}")
    print(f"   最终训练准确率: {final_acc:.4f}")
    print(f"   最终验证损失: {final_val_loss:.4f}")
    print(f"   最终验证准确率: {final_val_acc:.4f}")

    # 8. 模型预测
    print("\n7. 模型预测:")
    test_samples = X[:5]
    predictions = model.predict(test_samples, verbose=0)
    predicted_classes = (predictions > 0.5).astype(int).flatten()
    true_classes = y[:5].flatten()

    print("   前5个样本的预测结果:")
    for i in range(5):
        print(f"   样本{i+1}: 真实={true_classes[i]}, 预测={predicted_classes[i]}, 概率={predictions[i][0]:.4f}")

    # 9. 可视化训练过程
    print("\n8. 训练过程可视化:")
    try:
        plot_training_history(history, save_path='demo_training_history.png')
        print("   训练历史图表已保存为 'demo_training_history.png'")
    except Exception as e:
        print(f"   可视化失败: {e}")

    # 10. 资源使用情况
    print("\n9. 系统资源使用:")
    try:
        import psutil
        import GPUtil

        # CPU使用率
        cpu_percent = psutil.cpu_percent()
        memory = psutil.virtual_memory()

        print(f"   CPU使用率: {cpu_percent}%")
        print(f"   内存使用: {memory.percent}% ({memory.used/1024/1024/1024:.1f}GB/{memory.total/1024/1024/1024:.1f}GB)")

        # GPU信息（如果可用）
        try:
            gpus = GPUtil.getGPUs()
            for i, gpu in enumerate(gpus):
                print(f"   GPU {i}: {gpu.name}")
                print(f"     使用率: {gpu.load*100:.1f}%")
                print(f"     内存使用: {gpu.memoryUtil*100:.1f}%")
        except:
            print("   GPU信息不可用")
    except ImportError:
        print("   需要安装psutil和GPUtil来查看系统资源")

    print("\n" + "="*60)
    print("TensorFlow 2.0 演示完成！")
    print("\n接下来您可以:")
    print("1. 运行各个模块的示例文件")
    print("2. 查看 README.md 了解详细说明")
    print("3. 阅读 '运行指南.md' 获取更多帮助")
    print("4. 探索实用案例模块")
    print("="*60)


def interactive_demo():
    """
    交互式演示，让用户选择要运行的部分
    """
    print("\n" + "="*60)
    print("TensorFlow 2.0 交互式演示")
    print("="*60)

    options = """
    请选择要运行的演示:

    1. 基础张量操作
    2. 自动微分示例
    3. 简单神经网络训练
    4. 图像分类简介
    5. 文本处理简介
    6. 运行完整演示

    输入选项 (1-6) 或 'q' 退出:
    """

    while True:
        choice = input(options).strip()

        if choice.lower() == 'q':
            print("退出演示。再见！")
            break

        try:
            choice = int(choice)
            if choice == 1:
                demo_tensors()
            elif choice == 2:
                demo_gradients()
            elif choice == 3:
                demo_simple_nn()
            elif choice == 4:
                demo_image_classification()
            elif choice == 5:
                demo_text_processing()
            elif choice == 6:
                quick_demo()
                break
            else:
                print("无效选项，请输入1-6之间的数字")
        except ValueError:
            print("无效输入，请输入数字或'q'")


def demo_tensors():
    """张量操作演示"""
    print("\n--- 张量操作演示 ---")

    # 创建张量
    tensor1 = tf.constant([[1, 2], [3, 4]])
    tensor2 = tf.random.normal((2, 2))

    print(f"张量1:\n{tensor1}")
    print(f"张量2:\n{tensor2}")

    # 运算
    print(f"加法:\n{tf.add(tensor1, tensor2)}")
    print(f"乘法:\n{tf.matmul(tensor1, tensor2)}")


def demo_gradients():
    """自动微分演示"""
    print("\n--- 自动微分演示 ---")

    x = tf.Variable(3.0)

    with tf.GradientTape() as tape:
        y = x**2 + 2*x + 1

    gradient = tape.gradient(y, x)
    print(f"当x={x.numpy()}, y={y.numpy()}")
    print(f"导数dy/dx={gradient.numpy()}")


def demo_simple_nn():
    """简单神经网络演示"""
    print("\n--- 简单神经网络演示 ---")

    # 创建模型
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(10, activation='relu', input_shape=(5,)),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    # 创建数据
    X = tf.random.normal((100, 5))
    y = tf.random.uniform((100, 1))

    # 编译和训练
    model.compile(optimizer='adam', loss='binary_crossentropy')
    history = model.fit(X, y, epochs=5, verbose=0)

    print(f"模型训练完成，最终损失: {history.history['loss'][-1]:.4f}")


def demo_image_classification():
    """图像分类简介"""
    print("\n--- 图像分类简介 ---")
    print("要运行完整的图像分类演示，请执行:")
    print("python 05_Practical_Cases/image_classification.py")
    print("该模块包含CNN、迁移学习等内容。")


def demo_text_processing():
    """文本处理简介"""
    print("\n--- 文本处理简介 ---")
    print("要运行完整的文本处理演示，请执行:")
    print("python 05_Practical_Cases/nlp_text_classification.py")
    print("该模块包含词嵌入、RNN、LSTM等内容。")


if __name__ == "__main__":
    # 检查命令行参数
    if len(sys.argv) > 1 and sys.argv[1] == "--interactive":
        interactive_demo()
    else:
        quick_demo()
        print("\n要运行交互式演示，请使用: python demo.py --interactive")