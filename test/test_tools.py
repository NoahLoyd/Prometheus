# test_tools.py

from tools.calculator import CalculatorTool
from tools.note_tool import NoteTool
from tools.file_tool import FileTool
from tools.summarizer_tool import SummarizerTool
from tools.internet_tool import InternetTool

def test_calculator_tool():
    tool = CalculatorTool()
    try:
        # Valid input
        assert tool.run("2 + 2") == "4", "CalculatorTool valid input test failed"
        # Invalid input
        assert "Error" in tool.run("2 / 0"), "CalculatorTool invalid input test failed"
        print("CalculatorTool: PASS")
    except AssertionError as e:
        print(f"CalculatorTool: FAIL - {e}")

def test_note_tool():
    tool = NoteTool(notes_file="test_notes.txt")
    try:
        # Valid input
        assert tool.run("save: Test note") == "Note saved successfully.", "NoteTool valid input test failed"
        assert "Test note" in tool.run("list"), "NoteTool list test failed"
        # Invalid input
        assert "Invalid command" in tool.run("invalid command"), "NoteTool invalid input test failed"
        print("NoteTool: PASS")
    except AssertionError as e:
        print(f"NoteTool: FAIL - {e}")

def test_file_tool():
    tool = FileTool()
    try:
        # Valid input
        assert tool.run("write: test_file.txt: Test content") == "File 'test_file.txt' written successfully.", "FileTool write test failed"
        assert tool.run("read: test_file.txt") == "Test content", "FileTool read test failed"
        # Invalid input
        assert "Invalid command" in tool.run("invalid command"), "FileTool invalid input test failed"
        print("FileTool: PASS")
    except AssertionError as e:
        print(f"FileTool: FAIL - {e}")

def test_summarizer_tool():
    tool = SummarizerTool()
    try:
        # Valid input
        long_text = "This is a long text that needs to be summarized to fit within a short width."
        assert tool.run(long_text) == "This is a long text that needs to be summarized...", "SummarizerTool valid input test failed"
        # Invalid input
        assert "Error" in tool.run(""), "SummarizerTool invalid input test failed"
        print("SummarizerTool: PASS")
    except AssertionError as e:
        print(f"SummarizerTool: FAIL - {e}")

def test_internet_tool():
    tool = InternetTool()
    try:
        # Valid input (requires SERPAPI_API_KEY to be set in environment and internet connection)
        result = tool.run("Python programming")
        assert "Error" not in result and result.strip() != "", "InternetTool valid input test failed"
        # Invalid input
        assert "Error" in tool.run(""), "InternetTool invalid input test failed"
        print("InternetTool: PASS")
    except AssertionError as e:
        print(f"InternetTool: FAIL - {e}")

if __name__ == "__main__":
    print("Testing tools...\n")
    test_calculator_tool()
    test_note_tool()
    test_file_tool()
    test_summarizer_tool()
    test_internet_tool()
    print("\nTesting complete.")