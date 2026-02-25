"""
Full-Stack Benchmark Verification Script
"""
import pytest
import sys

def run():
    return pytest.main(['tests/', '-k', 'not test_qpu_execution', '-q'])

if __name__ == '__main__':
    sys.exit(run())
