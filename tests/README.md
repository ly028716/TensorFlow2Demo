# TensorFlow 2.0 学习项目单元测试

本目录包含项目所有模块的单元测试。

## 📁 测试文件结构

```
tests/
├── __init__.py                    # 测试模块初始化
├── test_01_basics.py              # 基础模块测试
├── test_02_neural_networks.py     # 神经网络模块测试
├── test_03_data_processing.py     # 数据处理模块测试
├── test_04_model_training.py      # 模型训练模块测试
├── test_06_utils.py               # 工具函数测试
└── README.md                      # 本文件
```

## 🚀 运行测试

### 方法一：使用测试运行脚本（推荐）

```bash
# 运行所有测试
python run_tests.py

# 运行指定模块的测试
python run_tests.py -m 01_basics
python run_tests.py -m 02_neural_networks
python run_tests.py -m 03_data_processing
python run_tests.py -m 04_model_training
python run_tests.py -m 06_utils

# 列出所有可用的测试模块
python run_tests.py -l

# 以简洁模式运行测试
python run_tests.py -v 1

# 以静默模式运行测试
python run_tests.py -v 0
```

### 方法二：使用unittest直接运行

```bash
# 运行所有测试
python -m unittest discover tests

# 运行单个测试文件
python -m unittest tests.test_01_basics

# 运行单个测试类
python -m unittest tests.test_01_basics.TestTensorFlowBasics

# 运行单个测试方法
python -m unittest tests.test_01_basics.TestTensorFlowBasics.test_tensor_creation
```

### 方法三：使用pytest运行（需要安装pytest）

```bash
# 安装pytest
pip install pytest pytest-cov

# 运行所有测试
pytest

# 运行指定文件的测试
pytest tests/test_01_basics.py

# 运行指定测试类
pytest tests/test_01_basics.py::TestTensorFlowBasics

# 运行指定测试方法
pytest tests/test_01_basics.py::TestTensorFlowBasics::test_tensor_creation

# 生成覆盖率报告
pytest --cov=. --cov-report=html

# 显示详细输出
pytest -v

# 显示打印输出
pytest -s
```

## 📊 测试覆盖范围

### test_01_basics.py - 基础模块测试

- ✅ 张量创建和操作
- ✅ 张量形状操作
- ✅ 变量创建和操作
- ✅ 自动微分
- ✅ NumPy转换
- ✅ tf.function装饰器
- ✅ 张量统计操作

**测试类：**
- `TestTensorFlowBasics`: 测试TensorFlow基础功能
- `TestTensorStatistics`: 测试张量统计操作

### test_02_neural_networks.py - 神经网络模块测试

- ✅ Sequential模型构建
- ✅ Functional API模型构建
- ✅ 模型子类化
- ✅ 各种网络层（Dense, Conv2D, Dropout等）
- ✅ 激活函数
- ✅ 自定义层

**测试类：**
- `TestKerasModels`: 测试Keras模型构建
- `TestLayers`: 测试神经网络层
- `TestActivationFunctions`: 测试激活函数
- `TestCustomLayers`: 测试自定义层

### test_03_data_processing.py - 数据处理模块测试

- ✅ tf.data API基础操作
- ✅ Dataset操作（take, skip, map, filter等）
- ✅ 批处理和混洗
- ✅ 数据预处理（归一化、标准化）
- ✅ One-hot编码
- ✅ 数据增强

**测试类：**
- `TestTFDataAPI`: 测试tf.data API
- `TestDataPreprocessing`: 测试数据预处理
- `TestDataAugmentation`: 测试数据增强

### test_04_model_training.py - 模型训练模块测试

- ✅ 模型编译
- ✅ 基本训练
- ✅ 带验证集的训练
- ✅ 回调函数（早停、模型检查点）
- ✅ 模型评估
- ✅ 模型预测
- ✅ 模型保存和加载

**测试类：**
- `TestModelCompilation`: 测试模型编译
- `TestModelTraining`: 测试模型训练
- `TestCallbacks`: 测试回调函数
- `TestModelEvaluation`: 测试模型评估
- `TestModelSaveLoad`: 测试模型保存和加载

### test_06_utils.py - 工具函数测试

- ✅ 参数统计
- ✅ 模型摘要
- ✅ 模型大小计算
- ✅ 训练历史可视化
- ✅ 数据处理工具
- ✅ 指标计算

**测试类：**
- `TestModelUtilities`: 测试模型工具函数
- `TestVisualizationTools`: 测试可视化工具
- `TestDataProcessingUtils`: 测试数据处理工具
- `TestMetricsCalculation`: 测试指标计算

## 📝 编写测试的最佳实践

### 1. 测试命名规范

```python
class TestFeatureName(unittest.TestCase):
    """测试某个功能"""
    
    def test_specific_behavior(self):
        """测试特定行为"""
        pass
```

### 2. 使用setUp和tearDown

```python
def setUp(self):
    """测试前的准备工作"""
    tf.random.set_seed(42)
    np.random.seed(42)

def tearDown(self):
    """测试后的清理工作"""
    tf.keras.backend.clear_session()
```

### 3. 断言使用

```python
# 相等性断言
self.assertEqual(a, b)
self.assertNotEqual(a, b)

# 真值断言
self.assertTrue(condition)
self.assertFalse(condition)

# 数值断言
self.assertAlmostEqual(a, b, places=5)
self.assertGreater(a, b)
self.assertLess(a, b)

# 数组断言（使用NumPy）
np.testing.assert_array_equal(a, b)
np.testing.assert_array_almost_equal(a, b, decimal=5)
```

### 4. 异常测试

```python
with self.assertRaises(ValueError):
    # 应该抛出ValueError的代码
    function_that_raises()
```

## 🔧 持续集成

测试可以集成到CI/CD流程中：

```yaml
# .github/workflows/tests.yml 示例
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Set up Python
        uses: actions/setup-python@v2
        with:
          python-version: 3.8
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install pytest pytest-cov
      - name: Run tests
        run: pytest --cov=. --cov-report=xml
```

## 📈 测试覆盖率

生成测试覆盖率报告：

```bash
# 使用pytest生成覆盖率报告
pytest --cov=. --cov-report=html

# 使用coverage.py
coverage run -m unittest discover tests
coverage report
coverage html
```

## 🐛 调试测试

```bash
# 运行测试时显示打印输出
python -m unittest tests.test_01_basics -v

# 使用pytest显示打印输出
pytest tests/test_01_basics.py -s

# 在失败时进入调试器
pytest --pdb
```

## ⚠️ 注意事项

1. **随机种子**：测试中使用固定的随机种子确保可重复性
2. **资源清理**：测试后清理临时文件和TensorFlow会话
3. **独立性**：每个测试应该独立运行，不依赖其他测试
4. **速度**：避免在单元测试中训练大型模型
5. **覆盖率**：目标是达到80%以上的代码覆盖率

## 📚 参考资源

- [unittest官方文档](https://docs.python.org/3/library/unittest.html)
- [pytest官方文档](https://docs.pytest.org/)
- [TensorFlow测试指南](https://www.tensorflow.org/guide/test)

---

**更新日期**: 2025-12-11
**版本**: 1.0.0

