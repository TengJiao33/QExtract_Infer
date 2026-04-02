# QExtract-Infer 测试 conftest
import pytest


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: 标记慢速测试")
    config.addinivalue_line("markers", "cuda: 需要 CUDA GPU")
