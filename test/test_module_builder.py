import unittest
import os
import tempfile
import shutil
from unittest.mock import patch, mock_open
from tools.module_builder import ModuleBuilderTool

class TestModuleBuilder(unittest.TestCase):
    def setUp(self):
        """Set up test environment with temporary directory."""
        self.temp_dir = tempfile.mkdtemp()
        self.builder = ModuleBuilderTool()
        
        # Create necessary subdirectories
        for dir_path in self.builder.MODULE_TYPE_DIRS.values():
            os.makedirs(os.path.join(self.temp_dir, dir_path), exist_ok=True)
        
        # Change to temp directory for tests
        self.original_dir = os.getcwd()
        os.chdir(self.temp_dir)
        
        # Create empty AddOnNotebook.log
        with open("AddOnNotebook.log", "w") as f:
            f.write("")

    def tearDown(self):
        """Clean up temporary directory after tests."""
        os.chdir(self.original_dir)
        shutil.rmtree(self.temp_dir)

    def test_write_module_legacy_plan(self):
        """Test writing a module with legacy plan format."""
        plan = {
            "file": "my_tool.py",
            "code": "class MyTool:\n    pass",
            "test": "def test_my_tool():\n    pass"
        }
        
        result = self.builder.write_module(plan)
        
        # Check if files were written to correct locations
        self.assertIn("tools/my_tool.py", result["written"])
        self.assertIn("tests/my_tool_test.py", result["written"])
        
        # Verify file contents
        with open("tools/my_tool.py") as f:
            content = f.read()
            self.assertIn('"""Tool Module: my_tool', content)
            self.assertIn("class MyTool:", content)

    def test_write_module_enhanced_plan(self):
        """Test writing modules with enhanced multi-file plan format."""
        plan = {
            "files": [
                {
                    "path": "validator_module.py",
                    "code": "def validate_something(): pass",
                    "type": "validator"
                },
                {
                    "path": "core_module.py",
                    "code": "class CoreModule: pass",
                    "type": "core"
                }
            ]
        }
        
        result = self.builder.write_module(plan)
        
        # Check correct routing
        self.assertIn("validators/validator_module.py", result["written"])
        self.assertIn("core/core_module.py", result["written"])
        
        # Verify headers
        with open("validators/validator_module.py") as f:
            content = f.read()
            self.assertIn('"""Validator Module:', content)
        
        with open("core/core_module.py") as f:
            content = f.read()
            self.assertIn('"""Core Module:', content)

    def test_overwrite_protection(self):
        """Test that files are not overwritten unless explicitly allowed."""
        # First write
        plan1 = {
            "file": "protected.py",
            "code": "# Version 1",
            "type": "tool"
        }
        self.builder.write_module(plan1)
        
        # Attempt overwrite
        plan2 = {
            "file": "protected.py",
            "code": "# Version 2",
            "type": "tool"
        }
        result = self.builder.write_module(plan2)
        
        self.assertIn("tools/protected.py", result["skipped"])
        
        # Check content remains unchanged
        with open("tools/protected.py") as f:
            self.assertIn("# Version 1", f.read())
        
        # Test explicit overwrite
        plan3 = {
            "file": "protected.py",
            "code": "# Version 3",
            "type": "tool",
            "overwrite_allowed": True
        }
        result = self.builder.write_module(plan3)
        
        self.assertIn("tools/protected.py", result["written"])
        with open("tools/protected.py") as f:
            self.assertIn("# Version 3", f.read())

    def test_snake_case_enforcement(self):
        """Test that filenames are converted to snake_case."""
        test_cases = [
            ("MyTestTool.py", "my_test_tool.py"),
            ("fastAPI_Tool.py", "fast_api_tool.py"),
            ("Quick-Brown-Fox.py", "quick_brown_fox.py"),
            ("SNAKEcase_TEST.py", "snakecase_test.py")
        ]
        
        for input_name, expected in test_cases:
            plan = {
                "file": input_name,
                "code": "pass",
                "type": "tool"
            }
            result = self.builder.write_module(plan)
            expected_path = os.path.join("tools", expected)
            self.assertIn(expected_path, result["written"])

    def test_module_type_routing(self):
        """Test that files are routed to correct directories based on type."""
        test_cases = [
            ("test_module.py", "test", "tests/"),
            ("validation_tool.py", "validator", "validators/"),
            ("core_engine.py", "core", "core/"),
            ("helper_tool.py", "tool", "tools/")
        ]
        
        for filename, mod_type, expected_dir in test_cases:
            plan = {
                "file": filename,
                "code": "pass",
                "type": mod_type
            }
            result = self.builder.write_module(plan)
            expected_path = os.path.join(expected_dir, self.builder._to_snake_case(filename))
            self.assertIn(expected_path, result["written"])

    def test_header_insertion(self):
        """Test that appropriate headers are inserted based on module type."""
        test_cases = [
            ("tool", "Tool Module"),
            ("test", "Test Module"),
            ("validator", "Validator Module"),
            ("core", "Core Module")
        ]
        
        for mod_type, expected_header in test_cases:
            plan = {
                "file": f"test_{mod_type}.py",
                "code": "# Some code",
                "type": mod_type
            }
            result = self.builder.write_module(plan)
            written_file = result["written"][0]
            
            with open(written_file) as f:
                content = f.read()
                self.assertIn(f'"""{expected_header}:', content)
                self.assertIn('Type: ' + mod_type, content)

    @patch('tools.module_builder.validate_security')
    def test_validator_integration(self, mock_validate):
        """Test that validators are called and results are logged."""
        # Setup mock validator
        mock_validate.return_value = (True, "Security check passed")
        
        plan = {
            "file": "secure_tool.py",
            "code": "class SecureTool: pass",
            "type": "tool"
        }
        
        result = self.builder.write_module(plan)
        
        # Check validator was called
        self.assertTrue(mock_validate.called)
        
        # Check logging
        with open("AddOnNotebook.log") as f:
            log_content = f.read()
            self.assertIn("VALIDATOR", log_content)
            self.assertIn("Security check passed", log_content)

    def test_invalid_plan_handling(self):
        """Test handling of invalid build plans."""
        invalid_plans = [
            {},  # Empty plan
            {"file": "test.py"},  # Missing code
            {"files": "not_a_list"},  # Invalid files format
            {"files": [{"path": "test.py"}]}  # Missing code in file entry
        ]
        
        for plan in invalid_plans:
            result = self.builder.write_module(plan)
            self.assertTrue(len(result["errors"]) > 0)
            self.assertEqual(len(result["written"]), 0)

    def test_type_inference(self):
        """Test module type inference from file paths and names."""
        test_cases = [
            ("tools/something.py", "tool"),
            ("validators/check.py", "validator"),
            ("core/engine.py", "core"),
            ("tests/test_module.py", "test"),
            ("my_tool_helper.py", "tool"),
            ("test_validator.py", "test"),
            ("core_module.py", "core"),
            ("random_file.py", None)
        ]
        
        for path, expected_type in test_cases:
            inferred_type = self.builder._infer_type_from_path(path)
            self.assertEqual(inferred_type, expected_type)

    def test_placeholder_test_generation(self):
        """Test generation of placeholder test files."""
        plan = {
            "file": "new_tool.py",
            "code": "class NewTool: pass",
            "type": "tool"
        }
        
        result = self.builder.write_module(plan)
        
        # Check that a placeholder test was generated
        test_file = "tests/new_tool_test.py"
        self.assertIn(test_file, result["written"])
        
        # Verify placeholder test contents
        with open(test_file) as f:
            content = f.read()
            self.assertIn("test_placeholder", content)
            self.assertIn("new_tool.py", content)

if __name__ == '__main__':
    unittest.main()
