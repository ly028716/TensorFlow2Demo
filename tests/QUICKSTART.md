# 测试快速入门

5分钟快速了解如何运行和使用项目测试。

## 🚀 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 运行第一个测试

```bash
# 运行基础模块测试
python run_tests.py -m 01_basics
```

预期输出：
```
==================================================
运行 01_basics 模块测试
==================================================
test_gradient_computation ... ok
test_numpy_conversion ... ok
test_reduce_operations ... ok
test_tensor_creation ... ok
test_tensor_operations ... ok
test_tensor_shape_operations ... ok
test_tf_function_decorator ... ok
test_variable_creation ... ok

----------------------------------------------------------------------
Ran 8 tests in 2.345s

OK
```

### 3. 运行所有测试

```bash
python run_tests.py
```

## 📊 查看测试覆盖率

```bash
# 安装pytest（如果还没安装）
pip install pytest pytest-cov

# 生成覆盖率报告
pytest --cov=. --cov-report=html

# 在浏览器中查看报告
# Windows: start htmlcov/index.html
# macOS: open htmlcov/index.html
# Linux: xdg-open htmlcov/index.html
```

## 🎯 测试文件说明

| 测试文件 | 测试内容 | 运行命令 |
|---------|---------|---------|
| test_01_basics.py | 张量、变量、自动微分 | `python run_tests.py -m 01_basics` |
| test_02_neural_networks.py | Keras API、网络层 | `python run_tests.py -m 02_neural_networks` |
| test_03_data_processing.py | 数据处理、tf.data | `python run_tests.py -m 03_data_processing` |
| test_04_model_training.py | 模型训练、评估 | `python run_tests.py -m 04_model_training` |
| test_06_utils.py | 工具函数 | `python run_tests.py -m 06_utils` |
| test_integration.py | 端到端测试 | `python run_tests.py -m integration` |

## 💡 常用命令

```bash
# 列出所有测试模块
python run_tests.py -l

# 详细输出
python run_tests.py -v 2

# 简洁输出
python run_tests.py -v 1

# 使用pytest运行特定测试
pytest tests/test_01_basics.py::TestTensorFlowBasics::test_tensor_creation

# 运行失败的测试
pytest --lf

# 并行运行（需要安装pytest-xdist）
pip install pytest-xdist
pytest -n auto
```

## 🔍 理解测试输出

### 成功的测试
```
test_tensor_creation ... ok
```
✅ 表示测试通过

### 失败的测试
```
test_tensor_creation ... FAIL
```
❌ 表示测试失败，会显示详细错误信息

### 错误的测试
```
test_tensor_creation ... ERROR
```
⚠️ 表示测试执行出错（通常是代码错误）

### 跳过的测试
```
test_tensor_creation ... skipped 'reason'
```
⏭️ 表示测试被跳过

## 📖 下一步

- 📚 阅读 [tests/README.md](README.md) 了解详细测试说明
- 📖 查看 [测试指南.md](../测试指南.md) 学习如何编写测试
- 🔍 浏览测试文件学习测试写法
- ✍️ 为新功能编写测试

## ❓ 遇到问题？

1. 确保在项目根目录运行命令
2. 检查是否安装了所有依赖
3. 查看 [测试指南.md](../测试指南.md) 的常见问题部分
4. 运行 `python -m unittest tests.test_01_basics -v` 获取详细输出

---

祝测试顺利！🎉

