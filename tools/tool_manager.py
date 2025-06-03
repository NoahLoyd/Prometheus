# tools/tool_manager.py

from typing import Dict, Type, Optional
from tools.base_tool import BaseTool
from addons.notebook import AddOnNotebook

class ToolManager:
    """
    Manages registration and invocation of modular tools for the AGI system.
    """

    def __init__(self, notebook: Optional[AddOnNotebook] = None):
        self.tools: Dict[str, BaseTool] = {}
        self.notebook = notebook or AddOnNotebook()
        
        if self.notebook:
            self.notebook.log("tool_manager", "INITIALIZATION", "ToolManager initialized", metadata={"tools_count": 0})

    def register_tool(self, tool_instance: BaseTool):
        """
        Registers a tool by its .name attribute (lowercased).
        """
        if not hasattr(tool_instance, 'name') or not isinstance(tool_instance.name, str):
            error_msg = f"Tool {tool_instance} must have a string 'name' attribute."
            if self.notebook:
                self.notebook.log("tool_manager", "REGISTRATION_ERROR", error_msg, metadata={
                    "tool_instance": str(tool_instance),
                    "has_name_attr": hasattr(tool_instance, 'name'),
                    "name_type": type(getattr(tool_instance, 'name', None)).__name__
                })
            raise ValueError(error_msg)
        
        name = tool_instance.name.strip().lower()
        if not name:
            error_msg = "Tool name must be a non-empty string."
            if self.notebook:
                self.notebook.log("tool_manager", "REGISTRATION_ERROR", error_msg, metadata={
                    "tool_instance": str(tool_instance),
                    "original_name": tool_instance.name
                })
            raise ValueError(error_msg)
        
        # Check for existing tool with same name
        if name in self.tools:
            if self.notebook:
                self.notebook.log("tool_manager", "TOOL_OVERWRITE", f"Overwriting existing tool: {name}", metadata={
                    "tool_name": name,
                    "previous_tool": str(self.tools[name]),
                    "new_tool": str(tool_instance)
                })
        
        self.tools[name] = tool_instance
        
        if self.notebook:
            self.notebook.log("tool_manager", "TOOL_REGISTERED", f"Tool registered successfully: {name}", metadata={
                "tool_name": name,
                "original_name": tool_instance.name,
                "tool_class": tool_instance.__class__.__name__,
                "tool_instance": str(tool_instance),
                "total_tools": len(self.tools)
            })

    def call_tool(self, name: str, query: str) -> str:
        """
        Calls a registered tool by name with the provided query.
        """
        original_name = name
        name = name.strip().lower()
        
        if self.notebook:
            self.notebook.log("tool_manager", "TOOL_CALL_START", f"Attempting to call tool: {name}", metadata={
                "tool_name": name,
                "original_name": original_name,
                "query": query,
                "query_length": len(query),
                "available_tools": list(self.tools.keys())
            })
        
        tool = self.tools.get(name)
        if not tool:
            # Optionally, add aliases here if necessary
            available = ', '.join(self.tools.keys())
            error_msg = f"Tool '{name}' not found. Available tools: {available if available else 'none'}"
            
            if self.notebook:
                self.notebook.log("tool_manager", "TOOL_NOT_FOUND", error_msg, metadata={
                    "requested_tool": name,
                    "original_name": original_name,
                    "available_tools": list(self.tools.keys()),
                    "available_count": len(self.tools)
                })
            
            return error_msg
        
        try:
            if self.notebook:
                self.notebook.log("tool_manager", "TOOL_EXECUTION_START", f"Executing tool: {name}", metadata={
                    "tool_name": name,
                    "tool_class": tool.__class__.__name__,
                    "query": query
                })
            
            result = tool.run(query)
            
            if self.notebook:
                self.notebook.log("tool_manager", "TOOL_EXECUTION_SUCCESS", f"Tool executed successfully: {name}", metadata={
                    "tool_name": name,
                    "tool_class": tool.__class__.__name__,
                    "query": query,
                    "result": result,
                    "result_length": len(str(result))
                })
            
            return result
            
        except Exception as e:
            error_msg = f"Error running tool '{name}': {e}"
            
            if self.notebook:
                self.notebook.log("tool_manager", "TOOL_EXECUTION_ERROR", error_msg, metadata={
                    "tool_name": name,
                    "tool_class": tool.__class__.__name__,
                    "query": query,
                    "error": str(e),
                    "error_type": type(e).__name__
                })
            
            return error_msg

    def get_available_tools(self) -> Dict[str, str]:
        """
        Returns a dictionary of available tools and their descriptions.
        """
        tools_info = {}
        for name, tool in self.tools.items():
            description = getattr(tool, 'description', 'No description available')
            tools_info[name] = description
        
        if self.notebook:
            self.notebook.log("tool_manager", "TOOLS_LISTED", f"Listed {len(tools_info)} available tools", metadata={
                "tools_count": len(tools_info),
                "tool_names": list(tools_info.keys())
            })
        
        return tools_info

    def remove_tool(self, name: str) -> bool:
        """
        Removes a tool from the registry.
        Returns True if tool was removed, False if tool was not found.
        """
        name = name.strip().lower()
        
        if name in self.tools:
            removed_tool = self.tools.pop(name)
            if self.notebook:
                self.notebook.log("tool_manager", "TOOL_REMOVED", f"Tool removed: {name}", metadata={
                    "tool_name": name,
                    "removed_tool": str(removed_tool),
                    "remaining_tools": len(self.tools)
                })
            return True
        else:
            if self.notebook:
                self.notebook.log("tool_manager", "TOOL_REMOVE_FAILED", f"Tool not found for removal: {name}", metadata={
                    "requested_tool": name,
                    "available_tools": list(self.tools.keys())
                })
            return False

    def clear_tools(self):
        """
        Removes all tools from the registry.
        """
        tool_count = len(self.tools)
        tool_names = list(self.tools.keys())
        
        self.tools.clear()
        
        if self.notebook:
            self.notebook.log("tool_manager", "TOOLS_CLEARED", f"All tools cleared from registry", metadata={
                "cleared_count": tool_count,
                "cleared_tools": tool_names
            })

    def get_tool_count(self) -> int:
        """
        Returns the number of registered tools.
        """
        count = len(self.tools)
        
        if self.notebook:
            self.notebook.log("tool_manager", "TOOL_COUNT_REQUESTED", f"Tool count requested: {count}", metadata={
                "tool_count": count,
                "tool_names": list(self.tools.keys())
            })
        
        return count
