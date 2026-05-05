"""
此测试用于统一且自动化地覆盖 `examples/` 目录下的所有独立 Python 脚本。
它能在后续每次运行 CI 时，确保为初学者提供的 Demo 脚本依然能够完美跑通，
避免后续 PyPOTS 版本更新导致示例代码报错失效。
"""

import os
import glob
import subprocess
import pytest

from pypots.utils.logging import logger

# 1. 扫描 examples 文件夹下的所有以 `.py` 结尾的文件
# 并排除那些并非真正示例文件的系统或隐藏文件
examples_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../examples"))
example_scripts = glob.glob(os.path.join(examples_dir, "**/*.py"), recursive=True)

# 筛选出需要运行测试的合法文件（比如排除 __init__.py 或者临时文件）
valid_scripts = [
    script for script in example_scripts
    if not os.path.basename(script).startswith("__") and not "checkpoint" in script
]


@pytest.mark.parametrize("script_path", valid_scripts)
def test_standalone_examples_can_run(script_path):
    """
    我们将每一个收集到的示例脚本作为独立的子进程执行。
    如果子进程正常退出 (退出码 exit code == 0)，说明示例代码有效且正确。
    反之则直接让单元测试爆出错误，提醒开发者修复对应 Example。
    """
    script_name = os.path.relpath(script_path, examples_dir)
    logger.info(f"🚀 [Testing Example Script] 正在以子进程方式运行: {script_name}...")

    # 执行命令，这里设定较长的 timeout 确保哪怕大点的例子能训得完，或者在示例里用更小的 epoch。
    # 我们期望每个示例脚本都能像小白用户执行 `python example.py` 那样正常运转。
    result = subprocess.run(
        ["python", script_path],
        capture_output=True,
        text=True,
        timeout=180
    )

    # 验证子进程退出是否成功
    if result.returncode != 0:
        logger.error(f"❌ '{script_name}' 运行失败!\n\nStandard Output:\n{result.stdout}\n\nStandard Error:\n{result.stderr}")
        pytest.fail(f"示例代码 {script_name} 执行报错，请检查它是否与框架的最新 API 或参数不兼容。")

    logger.info(f"✅ '{script_name}' 成功跑通！")

if __name__ == "__main__":
    pytest.main(["-s", __file__])

