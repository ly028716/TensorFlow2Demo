"""
TensorFlow 2.0 学习项目测试运行脚本

运行所有单元测试或指定模块的测试
"""

import unittest
import sys
import os
import argparse

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def run_all_tests(verbosity=2):
    """
    运行所有测试

    Args:
        verbosity: 输出详细程度 (0-2)
    """
    print("=" * 60)
    print("运行所有单元测试")
    print("=" * 60)

    # 发现并运行所有测试
    loader = unittest.TestLoader()
    start_dir = "tests"
    suite = loader.discover(start_dir, pattern="test_*.py")

    runner = unittest.TextTestRunner(verbosity=verbosity)
    result = runner.run(suite)

    # 打印测试摘要
    print("\n" + "=" * 60)
    print("测试摘要")
    print("=" * 60)
    print(f"运行测试数: {result.testsRun}")
    print(f"成功: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"失败: {len(result.failures)}")
    print(f"错误: {len(result.errors)}")
    print(f"跳过: {len(result.skipped)}")
    print("=" * 60)

    return result.wasSuccessful()


def run_module_tests(module_name, verbosity=2):
    """
    运行指定模块的测试

    Args:
        module_name: 模块名称 (例如: '01_basics', '02_neural_networks')
        verbosity: 输出详细程度 (0-2)
    """
    print("=" * 60)
    print(f"运行 {module_name} 模块测试")
    print("=" * 60)

    test_file = f"tests/test_{module_name}.py"

    if not os.path.exists(test_file):
        print(f"错误: 测试文件 {test_file} 不存在")
        return False

    # 加载指定模块的测试
    loader = unittest.TestLoader()
    suite = loader.discover("tests", pattern=f"test_{module_name}.py")

    runner = unittest.TextTestRunner(verbosity=verbosity)
    result = runner.run(suite)

    return result.wasSuccessful()


def list_test_modules():
    """列出所有可用的测试模块"""
    print("=" * 60)
    print("可用的测试模块")
    print("=" * 60)

    test_files = [f for f in os.listdir("tests") if f.startswith("test_") and f.endswith(".py")]

    for i, test_file in enumerate(test_files, 1):
        module_name = test_file.replace("test_", "").replace(".py", "")
        print(f"{i}. {module_name}")

    print("=" * 60)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="TensorFlow 2.0 学习项目测试运行器",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python run_tests.py                    # 运行所有测试
  python run_tests.py -m 01_basics       # 运行基础模块测试
  python run_tests.py -l                 # 列出所有测试模块
  python run_tests.py -v 1               # 以简洁模式运行所有测试
        """,
    )

    parser.add_argument(
        "-m", "--module", type=str, help="指定要测试的模块名称 (例如: 01_basics, 02_neural_networks)"
    )

    parser.add_argument("-l", "--list", action="store_true", help="列出所有可用的测试模块")

    parser.add_argument(
        "-v",
        "--verbosity",
        type=int,
        choices=[0, 1, 2],
        default=2,
        help="输出详细程度: 0=静默, 1=简洁, 2=详细 (默认: 2)",
    )

    args = parser.parse_args()

    # 列出测试模块
    if args.list:
        list_test_modules()
        return 0

    # 运行指定模块的测试
    if args.module:
        success = run_module_tests(args.module, args.verbosity)
    else:
        # 运行所有测试
        success = run_all_tests(args.verbosity)

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
