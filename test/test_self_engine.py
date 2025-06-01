import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, call

# Assuming these are your actual imports - adjust as needed
from promethyn.engine.self_coding import SelfCodingEngine
from promethyn.tools.prompt_decomposer import PromptDecomposer
from promethyn.tools.module_builder import ModuleBuilderTool
from promethyn.validators import (
    PlanVerifier,
    MathEvaluator,
    TestToolRunner,
    SecurityValidator
)
from promethyn.notebook import AddOnNotebook

@pytest.fixture
def temp_test_dir():
    """Create and clean up a temporary test directory."""
    test_dir = Path(tempfile.mkdtemp())
    yield test_dir
    shutil.rmtree(test_dir)

@pytest.fixture
def mock_plan():
    """Create a mock plan with type: tool."""
    return {
        'type': 'tool',
        'name': 'test_tool',
        'description': 'A test tool for testing',
        'implementation': {
            'language': 'python',
            'code': 'def execute(): return True'
        }
    }

@pytest.fixture
def mock_validators():
    """Create mock instances of all validators."""
    return {
        'plan_verifier': Mock(spec=PlanVerifier),
        'math_evaluator': Mock(spec=MathEvaluator),
        'test_runner': Mock(spec=TestToolRunner),
        'security_validator': Mock(spec=SecurityValidator)
    }

@pytest.fixture
def mock_notebook():
    """Create a mock AddOnNotebook instance."""
    return Mock(spec=AddOnNotebook)

@pytest.fixture
def engine(temp_test_dir, mock_validators, mock_notebook):
    """Create a SelfCodingEngine instance with mock components."""
    with patch('promethyn.engine.self_coding.PromptDecomposer') as mock_decomposer, \
         patch('promethyn.engine.self_coding.ModuleBuilderTool') as mock_builder:
        
        engine = SelfCodingEngine(
            working_dir=temp_test_dir,
            prompt_decomposer=mock_decomposer,
            module_builder=mock_builder,
            plan_verifier=mock_validators['plan_verifier'],
            math_evaluator=mock_validators['math_evaluator'],
            test_runner=mock_validators['test_runner'],
            security_validator=mock_validators['security_validator'],
            notebook=mock_notebook
        )
        yield engine

def test_successful_pipeline(engine, mock_plan, mock_validators, mock_notebook):
    """Test successful execution of the entire pipeline."""
    # Configure all validators to return success
    for validator in mock_validators.values():
        validator.validate.return_value = {'success': True, 'message': 'Passed'}

    # Mock the prompt decomposer response
    engine.prompt_decomposer.decompose.return_value = mock_plan
    
    # Mock the module builder response
    engine.module_builder.generate.return_value = {
        'success': True,
        'code': 'def test_tool(): return True'
    }

    # Execute the pipeline
    result = engine.process_prompt("Create a test tool")

    # Verify the pipeline execution order
    assert result['success'] is True
    
    # Verify PromptDecomposer was called
    engine.prompt_decomposer.decompose.assert_called_once()
    
    # Verify ModuleBuilder was called with the plan
    engine.module_builder.generate.assert_called_once_with(mock_plan)
    
    # Verify validators were called in sequence
    validation_sequence = [
        mock_validators['plan_verifier'].validate,
        mock_validators['math_evaluator'].validate,
        mock_validators['test_runner'].validate,
        mock_validators['security_validator'].validate
    ]
    
    for validator_call in validation_sequence:
        validator_call.assert_called_once()

    # Verify logging to notebook
    assert mock_notebook.log.call_count >= len(validation_sequence)

def test_validator_failure(engine, mock_plan, mock_validators, mock_notebook):
    """Test pipeline failure when a validator fails."""
    # Configure first validator to fail
    mock_validators['plan_verifier'].validate.return_value = {
        'success': False,
        'message': 'Plan validation failed'
    }

    # Mock successful prompt decomposition
    engine.prompt_decomposer.decompose.return_value = mock_plan

    # Execute the pipeline
    result = engine.process_prompt("Create a test tool")

    # Verify failure
    assert result['success'] is False
    assert 'Plan validation failed' in result.get('message', '')

    # Verify pipeline stopped after first validator
    mock_validators['math_evaluator'].validate.assert_not_called()
    mock_validators['test_runner'].validate.assert_not_called()
    mock_validators['security_validator'].validate.assert_not_called()

def test_invalid_plan_type(engine, mock_validators, mock_notebook):
    """Test handling of invalid plan type."""
    invalid_plan = {
        'type': 'invalid',
        'name': 'test_tool'
    }
    
    engine.prompt_decomposer.decompose.return_value = invalid_plan

    result = engine.process_prompt("Create an invalid tool")

    assert result['success'] is False
    assert 'Invalid plan type' in result.get('message', '')

def test_notebook_logging(engine, mock_plan, mock_validators, mock_notebook):
    """Test proper logging to AddOnNotebook."""
    # Configure successful validation
    for validator in mock_validators.values():
        validator.validate.return_value = {'success': True, 'message': 'Passed'}

    engine.prompt_decomposer.decompose.return_value = mock_plan
    engine.module_builder.generate.return_value = {
        'success': True,
        'code': 'def test_tool(): return True'
    }

    engine.process_prompt("Create a test tool")

    # Verify logging calls
    expected_log_calls = [
        call("Starting prompt decomposition"),
        call("Plan created successfully"),
        call("Starting validation pipeline"),
        call("Plan verification passed"),
        call("Math evaluation passed"),
        call("Test execution passed"),
        call("Security validation passed"),
        call("Tool registration successful")
    ]

    mock_notebook.log.assert_has_calls(expected_log_calls, any_order=True)

def test_tool_registration(engine, mock_plan, mock_validators):
    """Test successful tool registration after validation."""
    # Configure successful validation
    for validator in mock_validators.values():
        validator.validate.return_value = {'success': True, 'message': 'Passed'}

    engine.prompt_decomposer.decompose.return_value = mock_plan
    engine.module_builder.generate.return_value = {
        'success': True,
        'code': 'def test_tool(): return True'
    }

    result = engine.process_prompt("Create a test tool")

    assert result['success'] is True
    assert result.get('tool_registered') is True
    assert Path(engine.working_dir / 'test_tool.py').exists()

@pytest.mark.parametrize('validator_name', [
    'plan_verifier',
    'math_evaluator',
    'test_runner',
    'security_validator'
])
def test_individual_validator_failures(engine, mock_plan, mock_validators, 
                                    mock_notebook, validator_name):
    """Test failure cases for each validator individually."""
    # Configure all validators for success except the target one
    for name, validator in mock_validators.items():
        if name == validator_name:
            validator.validate.return_value = {
                'success': False,
                'message': f'{validator_name} validation failed'
            }
        else:
            validator.validate.return_value = {'success': True, 'message': 'Passed'}

    engine.prompt_decomposer.decompose.return_value = mock_plan
    engine.module_builder.generate.return_value = {
        'success': True,
        'code': 'def test_tool(): return True'
    }

    result = engine.process_prompt("Create a test tool")

    assert result['success'] is False
    assert validator_name in result.get('message', '').lower()
    assert mock_notebook.log.called

def test_working_directory_creation(temp_test_dir):
    """Test working directory is created and managed properly."""
    test_path = temp_test_dir / 'self_coding_test'
    
    with patch('promethyn.engine.self_coding.PromptDecomposer'), \
         patch('promethyn.engine.self_coding.ModuleBuilderTool'), \
         patch('promethyn.engine.self_coding.AddOnNotebook'):
        
        engine = SelfCodingEngine(working_dir=test_path)
        
        assert test_path.exists()
        assert test_path.is_dir()

def test_cleanup_on_error(engine, mock_plan, mock_validators):
    """Test cleanup of temporary files on error."""
    # Force an error in the middle of processing
    mock_validators['math_evaluator'].validate.side_effect = Exception("Unexpected error")

    engine.prompt_decomposer.decompose.return_value = mock_plan
    engine.module_builder.generate.return_value = {
        'success': True,
        'code': 'def test_tool(): return True'
    }

    with pytest.raises(Exception):
        engine.process_prompt("Create a test tool")

    # Verify temporary files are cleaned up
    temp_files = list(Path(engine.working_dir).glob('*.tmp'))
    assert len(temp_files) == 0
