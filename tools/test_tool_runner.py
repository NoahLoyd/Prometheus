"""
Module: promethyn.test_tool_runner

Elite, production-grade TestToolRunner for Promethyn’s self-coding system.

This module defines the TestToolRunner class, responsible for dynamically importing and executing test files using either pytest or unittest, returning structured results. The runner is robust, extensible, and follows Promethyn’s engineering standards, ensuring any generated test file is validated before installation.

Author: Promethyn AI
Python Version: 3.11+
"""

import importlib.util
import sys
import traceback
import types
import logging
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger("promethyn.test_tool_runner")
logger.setLevel(logging.INFO)


class TestToolRunner:
    """
    Runs Python test files dynamically, using pytest or unittest, and reports structured results.
    
    Features:
        - Dynamically imports test files
        - Executes tests with pytest or unittest (whichever is available)
        - Reports pass/fail status, errors, and logs
        - Handles import/runtime errors gracefully
        - Modular and ready for Promethyn system integration
    """

    def __init__(self) -> None:
        """Initializes the TestToolRunner."""
        self.pytest = self._import_optional("pytest")
        self.unittest = self._import_optional("unittest")

    def _import_optional(self, module_name: str) -> Optional[types.ModuleType]:
        """Safely attempts to import an optional module."""
        try:
            return importlib.import_module(module_name)
        except ImportError:
            return None

    def run_test_file(self, test_file_path: str) -> Dict[str, Any]:
        """
        Executes a test file and returns a structured result.

        Args:
            test_file_path (str): The path to the test file.

        Returns:
            Dict[str, Any]: Structured result including:
                - file (str): Path to the test file
                - framework (str): Framework used (pytest/unittest/none)
                - passed (bool): True if all tests passed, False otherwise
                - error_log (str): Any error, stack trace, or logs
                - details (Any): Framework-specific result details
        """
        abs_path = str(Path(test_file_path).resolve())
        result = {
            "file": abs_path,
            "framework": None,
            "passed": False,
            "error_log": "",
            "details": None,
        }

        if not Path(abs_path).is_file():
            result["error_log"] = f"File not found: {abs_path}"
            logger.error(result["error_log"])
            return result

        # Attempt to import the test file as a module
        module_name = f"_promethyn_gen_test_{Path(abs_path).stem}"
        try:
            spec = importlib.util.spec_from_file_location(module_name, abs_path)
            if spec is None or spec.loader is None:
                raise ImportError(f"Could not create module spec for {abs_path}")

            test_module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = test_module
            spec.loader.exec_module(test_module)
        except Exception as e:
            error_msg = f"ImportError in '{abs_path}': {e}\n{traceback.format_exc()}"
            result["error_log"] = error_msg
            logger.error(error_msg)
            return result

        # Try pytest first if available
        if self.pytest:
            logger.info(f"Testing '{abs_path}' with pytest")
            result["framework"] = "pytest"
            try:
                # pytest.main returns 0 for all tests passed, non-zero otherwise
                pytest_result = self.pytest.main(
                    [abs_path, "--tb=short", "-q", "--disable-warnings"],
                    plugins=[]
                )
                result["passed"] = pytest_result == 0
                result["details"] = f"pytest exit code: {pytest_result}"
            except Exception as e:
                error_msg = f"Pytest execution failed: {e}\n{traceback.format_exc()}"
                result["error_log"] = error_msg
                logger.error(error_msg)
            return result

        # Fallback to unittest if available
        if self.unittest:
            logger.info(f"Testing '{abs_path}' with unittest")
            result["framework"] = "unittest"
            try:
                import io
                import contextlib

                # Capture output for logging
                output = io.StringIO()
                loader = self.unittest.TestLoader()
                suite = loader.discover(str(Path(abs_path).parent), pattern=Path(abs_path).name)
                runner = self.unittest.TextTestRunner(stream=output, verbosity=2)
                test_result = runner.run(suite)
                result["passed"] = test_result.wasSuccessful()
                result["details"] = {
                    "testsRun": test_result.testsRun,
                    "failures": len(test_result.failures),
                    "errors": len(test_result.errors),
                }
                result["error_log"] = output.getvalue()
            except Exception as e:
                error_msg = f"Unittest execution failed: {e}\n{traceback.format_exc()}"
                result["error_log"] = error_msg
                logger.error(error_msg)
            return result

        # No test framework found
        error_msg = (
            "No supported test framework (pytest or unittest) found in environment. "
            "Cannot run test file."
        )
        result["framework"] = "none"
        result["error_log"] = error_msg
        logger.error(error_msg)
        return result

    @staticmethod
    def check_frameworks() -> Dict[str, bool]:
        """
        Checks which test frameworks are available.

        Returns:
            Dict[str, bool]: {'pytest': bool, 'unittest': bool}
        """
        return {
            "pytest": importlib.util.find_spec("pytest") is not None,
            "unittest": importlib.util.find_spec("unittest") is not None,
        }


# Example usage:
# runner = TestToolRunner()
# outcome = runner.run_test_file("/path/to/generated_test_file.py")
# print(outcome)
