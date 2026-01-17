# TensorFlow 2.0 学习项目

这是一个全面学习TensorFlow 2.0语法知识的项目，包括基础语法讲解、代码示例和实用案例。本项目旨在帮助开发者从基础到进阶系统地掌握TensorFlow 2.0的核心概念和实际应用。

## 📚 项目结构

```
TensorFlow2Demo/
├── 01_Basics/                     # 基础语法知识
│   ├── tensorflow_basics.py        # 张量、变量、自动微分等基础
│   └── eager_execution_demo.py    # 即时执行演示
│
├── 02_Neural_Networks/            # 神经网络基础
│   ├── neural_networks_basics.py  # 层、激活函数、损失函数等
│   └── custom_layers_models.py    # 自定义层和模型
│
├── 03_Data_Processing/             # 数据处理
│   ├── data_processing_basics.py  # 数据处理基础
│   └── tfdataset_examples.py      # tf.data API实用示例
│
├── 04_Model_Training/             # 模型训练与评估
│   └── model_training_basics.py   # 训练、评估、保存和加载
│
├── 05_Practical_Cases/            # 实用案例
│   ├── image_classification.py    # 图像分类案例
│   └── nlp_text_classification.py# 文本分类案例
│
├── data/                          # 数据存储目录
├── models/                        # 模型存储目录
├── notebooks/                     # Jupyter笔记本
├── 06_Utils/                      # 工具函数
├── tests/                         # 单元测试
├── requirements.txt               # 项目依赖
├── run_tests.py                   # 测试运行脚本
├── 测试指南.md                    # 测试文档
└── README.md                      # 项目说明
```

## 🚀 快速开始

### 环境要求

- Python 3.7+
- TensorFlow 2.13.0
- NumPy 1.24.3+
- matplotlib 3.7.2+
- pandas 2.0.3+
- scikit-learn 1.3.0+

### 安装依赖

```bash
pip install -r requirements.txt
```

或者手动安装主要依赖：

```bash
pip install tensorflow numpy matplotlib pandas scikit-learn jupyter Pillow
```

### 运行测试

项目包含完整的单元测试，确保代码质量：

```bash
# 运行所有测试
python run_tests.py

# 运行指定模块测试
python run_tests.py -m 01_basics

# 查看测试覆盖率
pytest --cov=. --cov-report=html
```

详细测试说明请查看 [测试指南.md](测试指南.md)

## 📖 学习路径

### 1. 基础语法知识 (01_Basics/)

从TensorFlow 2.0的基础概念开始学习：

- **张量(Tensors)**: 学习TensorFlow的基本数据结构
- **即时执行(Eager Execution)**: 理解TensorFlow 2.0的执行模式
- **变量和自动微分**: 掌握变量定义和梯度计算
- **函数装饰器**: 了解tf.function的使用

运行示例：
```python
python 01_Basics/tensorflow_basics.py
python 01_Basics/eager_execution_demo.py
```

### 2. 神经网络基础 (02_Neural_Networks/)

学习构建神经网络的基本组件：

- **Keras API**: 掌握Sequential、Functional和Subclassing三种API
- **网络层**: 了解各种层的功能和用法
- **激活函数**: 学习不同激活函数的特点
- **损失函数和优化器**: 选择合适的损失函数和优化策略

运行示例：
```python
python 02_Neural_Networks/neural_networks_basics.py
python 02_Neural_Networks/custom_layers_models.py
```

### 3. 数据处理 (03_Data_Processing/)

掌握TensorFlow的数据处理能力：

- **tf.data API**: 高效的数据管道
- **数据预处理**: 标准化、归一化、编码等
- **数据增强**: 提高模型泛化能力
- **性能优化**: 并行处理、缓存和预取

运行示例：
```python
python 03_Data_Processing/data_processing_basics.py
python 03_Data_Processing/tfdataset_examples.py
```

### 4. 模型训练与评估 (04_Model_Training/)

学习完整的模型训练流程：

- **模型编译**: 优化器、损失函数和指标设置
- **训练技巧**: 回调函数、正则化、学习率调整
- **模型评估**: 准确率、精确率、召回率等指标
- **模型保存**: 保存和加载模型

运行示例：
```python
python 04_Model_Training/model_training_basics.py
```

### 5. 实用案例 (05_Practical_Cases/)

通过实际案例巩固所学知识：

- **图像分类**: CNN、迁移学习、ResNet等
- **文本分类**: 词嵌入、RNN、LSTM、CNN等

运行示例：
```python
python 05_Practical_Cases/image_classification.py
python 05_Practical_Cases/nlp_text_classification.py
```

## 💡 核心特性

### 1. 实用性

每个模块都包含理论讲解和实际代码示例，可以直接运行和学习。

### 2. 渐进式学习

从基础概念到高级应用，循序渐进地学习TensorFlow 2.0。

### 3. 最佳实践

包含代码优化、性能调优和实际开发中的最佳实践。

### 4. 丰富的示例

涵盖图像处理、文本处理、时间序列等多种应用场景。

## 🛠️ 工具和技巧

### 1. 性能优化

- 使用tf.data API构建高效数据管道
- 利用并行处理加速训练
- 合理使用缓存和预取

### 2. 调试技巧

- 即时执行便于调试
- TensorBoard可视化
- 自定义回调函数监控训练

### 3. 模型部署

- 模型保存和加载
- SavedModel格式
- 转换为TFLite（移动端）

## 🎯 学习目标

通过本项目学习，您将能够：

1. **掌握TensorFlow 2.0的核心概念**
2. **熟练使用Keras API构建模型**
3. **高效处理各种类型的数据**
4. **实现常见的深度学习任务**
5. **优化模型性能和训练过程**

## 📊 示例代码

以下是一个简单的TensorFlow 2.0示例：

```python
import tensorflow as tf

# 创建模型
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(10,)),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# 编译模型
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# 准备数据
import numpy as np
X = np.random.random((1000, 10))
y = np.random.randint(0, 2, (1000, 1))

# 训练模型
history = model.fit(X, y, epochs=10, batch_size=32, validation_split=0.2)
```

## 🧪 单元测试

本项目包含完整的单元测试，确保代码质量和功能正确性。

### 测试覆盖范围

- ✅ **基础模块测试**: 张量操作、变量、自动微分
- ✅ **神经网络测试**: Keras API、网络层、激活函数
- ✅ **数据处理测试**: tf.data API、数据预处理、数据增强
- ✅ **模型训练测试**: 编译、训练、评估、保存加载
- ✅ **工具函数测试**: 可视化、参数统计
- ✅ **集成测试**: 端到端工作流程

### 快速运行测试

```bash
# 运行所有测试
python run_tests.py

# 运行指定模块测试
python run_tests.py -m 01_basics
python run_tests.py -m 02_neural_networks

# 使用pytest运行
pytest

# 生成覆盖率报告
pytest --cov=. --cov-report=html
```

详细信息请参考 [测试指南.md](测试指南.md) 和 [tests/README.md](tests/README.md)

## 🔗 相关资源

- [TensorFlow 官方文档](https://www.tensorflow.org/)
- [Keras 官方文档](https://keras.io/)
- [TensorFlow 教程](https://www.tensorflow.org/tutorials)
- [TensorFlow Hub](https://tfhub.dev/)

## 📝 学习建议

1. **循序渐进**: 按照模块顺序学习，打好基础
2. **动手实践**: 运行每个示例，修改参数观察结果
3. **深入理解**: 不只是使用API，理解背后的原理
4. **项目实践**: 尝试将学到的知识应用到实际项目中

## 🤝 贡献

欢迎提交问题报告、改进建议或直接贡献代码！

## 📄 许可证

本项目采用MIT许可证，详见LICENSE文件。

---

**Happy Learning! 🎉**

开始您的TensorFlow 2.0学习之旅吧！