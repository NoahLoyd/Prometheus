import unittest
from unittest.mock import Mock, patch
import json
from tools.prompt_decomposer import PromptDecomposer, STANDARD_CONTEXT

class TestPromptDecomposer(unittest.TestCase):
    def setUp(self):
        """Initialize PromptDecomposer with a mock LLMRouter for testing."""
        self.decomposer = PromptDecomposer({"models": ["simulated"], "use_simulation": True})
        # Create a mock for the LLMRouter
        self.mock_llm = Mock()
        self.decomposer.llm = self.mock_llm

    def test_decompose_tool_type(self):
        """Test decompose() for a basic tool-type prompt."""
        # Prepare mock LLM response
        mock_plan = {
            "file": "tools/test_tool.py",
            "class": "TestTool",
            "code": "class TestTool:\n    pass",
            "test": "def test_tool():\n    pass"
        }
        self.mock_llm.generate.return_value = json.dumps(mock_plan)

        # Test decomposition
        result = self.decomposer.decompose("Create a tool that does X")
        
        # Verify LLM was called with correct context
        self.mock_llm.generate.assert_called_once()
        call_args = self.mock_llm.generate.call_args[0]
        self.assertTrue(call_args[0].startswith(STANDARD_CONTEXT))
        
        # Verify result structure
        self.assertEqual(result["file"], "tools/test_tool.py")
        self.assertEqual(result["class"], "TestTool")
        self.assertTrue("code" in result)
        self.assertTrue("test" in result)

    def test_decompose_memory_module(self):
        """Test decompose() for a memory module type."""
        result = self.decomposer.decompose("Create a memory module for storing conversation history")
        
        # Verify it creates a multi-file plan
        self.assertTrue("files" in result)
        self.assertEqual(result["module_type"], "memory")
        
        # Check files structure
        files = result["files"]
        self.assertEqual(len(files), 2)  # Main module and interface
        
        # Verify main module
        main_module = next(f for f in files if "interfaces" not in f["path"])
        self.assertTrue("memory" in main_module["path"])
        self.assertTrue("code" in main_module)
        self.assertTrue("test" in main_module)
        
        # Verify interface
        interface = next(f for f in files if "interfaces" in f["path"])
        self.assertTrue("I" in interface["class"])
        self.assertTrue("NotImplementedError" in interface["code"])

    def test_call_llm_success(self):
        """Test successful LLM call with valid response."""
        mock_response = {
            "file": "tools/test.py",
            "class": "Test",
            "code": "class Test:\n    pass",
            "test": "def test_test():\n    pass"
        }
        self.mock_llm.generate.return_value = json.dumps(mock_response)
        
        result = self.decomposer._call_llm("test prompt")
        
        self.assertEqual(result, mock_response)
        self.mock_llm.generate.assert_called_once()

    def test_call_llm_invalid_json(self):
        """Test LLM call with invalid JSON response."""
        self.mock_llm.generate.return_value = "invalid json"
        
        result = self.decomposer._call_llm("test prompt")
        
        self.assertEqual(result, {})

    def test_call_llm_non_dict_response(self):
        """Test LLM call returning non-dict JSON."""
        self.mock_llm.generate.return_value = json.dumps(["not", "a", "dict"])
        
        result = self.decomposer._call_llm("test prompt")
        
        self.assertEqual(result, {})

    def test_validate_and_fallback_valid_legacy_plan(self):
        """Test validation of valid legacy plan."""
        plan = {
            "file": "tools/test.py",
            "class": "Test",
            "code": "class Test:\n    pass",
            "test": "def test_test():\n    pass"
        }
        
        result = self.decomposer._validate_and_fallback(plan, "test prompt")
        
        self.assertEqual(result, plan)

    def test_validate_and_fallback_invalid_legacy_plan(self):
        """Test fallback with invalid legacy plan."""
        invalid_plan = {
            "file": "tools/test.py",
            # Missing required keys
        }
        
        result = self.decomposer._validate_and_fallback(invalid_plan, "test prompt")
        
        self.assertTrue("UndefinedModule" in result["class"])
        self.assertTrue("undefined_module.py" in result["file"])

    def test_validate_and_fallback_valid_multifile_plan(self):
        """Test validation of valid multi-file plan."""
        plan = {
            "files": [{
                "path": "modules/memory/test_memory.py",
                "class": "TestMemory",
                "code": "class TestMemory:\n    pass",
                "test": "def test_memory():\n    pass"
            }],
            "module_type": "memory",
            "plan_generated": True
        }
        
        result = self.decomposer._validate_and_fallback(plan, "test prompt")
        
        self.assertEqual(result, plan)

    def test_validate_and_fallback_invalid_multifile_plan(self):
        """Test fallback with invalid multi-file plan."""
        invalid_plan = {
            "files": [{
                "path": "modules/memory/test_memory.py",
                # Missing required keys
            }]
        }
        
        result = self.decomposer._validate_and_fallback(invalid_plan, "test prompt")
        
        self.assertTrue("UndefinedModule" in result["class"])

    def test_detect_module_type(self):
        """Test module type detection for different prompts."""
        test_cases = [
            ("Create a memory module", "memory"),
            ("Build an agent that can solve tasks", "agent"),
            ("Implement a planning system", "planner"),
            ("Create an analyzer for code", "analyzer"),
            ("Build a router for messages", "router"),
            ("Create a unit test for X", "test"),
            ("Add security validation", "validator"),
            ("Implement core logic", "core"),
            ("Create a new tool", "tool"),
        ]
        
        for prompt, expected_type in test_cases:
            with self.subTest(prompt=prompt):
                result = self.decomposer.detect_module_type(prompt)
                self.assertEqual(result, expected_type)

    def test_create_multifile_plan(self):
        """Test creation of multi-file plan for non-tool modules."""
        result = self.decomposer.create_multifile_plan(
            "memory",
            "Create a conversation memory module"
        )
        
        self.assertTrue("files" in result)
        self.assertEqual(result["module_type"], "memory")
        self.assertTrue(result["plan_generated"])
        
        # Verify files structure
        files = result["files"]
        self.assertTrue(any("interfaces" in f["path"] for f in files))
        
        # Check main module
        main_module = next(f for f in files if "interfaces" not in f["path"])
        self.assertTrue("run" in main_module["code"])
        self.assertTrue("test" in main_module)
        
        # Check that safe name generation works
        self.assertTrue(any(f["path"].endswith("memory_create_a_conversation_memory_module.py") 
                          for f in files))

if __name__ == '__main__':
    unittest.main()
