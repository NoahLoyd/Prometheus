import os
import tempfile
import shutil
import subprocess
import sys
import logging
import resource
import threading
from typing import Dict, Any, Optional
from core.utils.path_utils import safe_path_join
from addons.notebook import AddOnNotebook

# Promethyn logger setup — assumes logging config elsewhere
logger = logging.getLogger("promethyn.sandbox")


class SandboxTimeout(Exception):
    pass


class SandboxPool:
    """
    Controls concurrent access to sandbox resources using a semaphore.
    """
    def __init__(self, max_concurrent: int, notebook: Optional[AddOnNotebook] = None):
        self.semaphore = threading.Semaphore(max_concurrent)
        self.notebook = notebook
        logger.info(f"Initialized SandboxPool with {max_concurrent} concurrent slots")
        
        if self.notebook:
            self.notebook.log("sandbox_runner", "POOL_INITIALIZED", f"SandboxPool initialized with {max_concurrent} concurrent slots", metadata={
                "max_concurrent": max_concurrent
            })
    
    def acquire(self):
        """Acquire a sandbox slot"""
        acquired = self.semaphore.acquire(blocking=True)
        if acquired:
            logger.debug("Acquired sandbox slot")
            if self.notebook:
                self.notebook.log("sandbox_runner", "SLOT_ACQUIRED", "Acquired sandbox slot", metadata={
                    "slot_acquired": True
                })
        return acquired
    
    def release(self):
        """Release a sandbox slot"""
        self.semaphore.release()
        logger.debug("Released sandbox slot")
        if self.notebook:
            self.notebook.log("sandbox_runner", "SLOT_RELEASED", "Released sandbox slot", metadata={
                "slot_released": True
            })


# Global sandbox pool instance with 5 concurrent slots
sandbox_pool = SandboxPool(5)


def _set_resource_limits(cpu_time_limit: int, memory_limit_mb: int) -> None:
    """
    Set resource limits for the child process.
    """
    try:
        # CPU time limit (seconds)
        resource.setrlimit(resource.RLIMIT_CPU, (cpu_time_limit, cpu_time_limit))
        # Address space limit (bytes)
        memory_bytes = memory_limit_mb * 1024 * 1024
        resource.setrlimit(resource.RLIMIT_AS, (memory_bytes, memory_bytes))
    except Exception as e:
        logger.error(f"Failed to set resource limits: {e}")
        raise


def _run_with_limits(
    cmd: list[str],
    cwd: str,
    cpu_time_limit: int,
    memory_limit_mb: int,
    timeout: int,
    env: Optional[dict] = None,
    notebook: Optional[AddOnNotebook] = None,
) -> Dict[str, Any]:
    """
    Run the given command with resource limits and a timeout.
    Captures stdout, stderr, and return code.
    """
    result = {
        "success": False,
        "stdout": "",
        "stderr": "",
        "exit_code": None,
    }

    if notebook:
        notebook.log("sandbox_runner", "EXECUTION_START", f"Starting sandboxed execution: {' '.join(cmd)}", metadata={
            "command": cmd,
            "working_directory": cwd,
            "cpu_time_limit": cpu_time_limit,
            "memory_limit_mb": memory_limit_mb,
            "timeout": timeout,
            "environment": env
        })

    def target(proc_result: dict):
        try:
            proc = subprocess.Popen(
                cmd,
                cwd=cwd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                stdin=subprocess.DEVNULL,
                preexec_fn=lambda: _set_resource_limits(cpu_time_limit, memory_limit_mb),
                env=env,
                close_fds=True,
            )
            try:
                out, err = proc.communicate(timeout=timeout)
                proc_result["stdout"] = out.decode("utf-8", errors="replace")
                proc_result["stderr"] = err.decode("utf-8", errors="replace")
                proc_result["exit_code"] = proc.returncode
                proc_result["success"] = proc.returncode == 0
                
                if notebook:
                    notebook.log("sandbox_runner", "PROCESS_COMPLETED", f"Process completed with exit code: {proc.returncode}", metadata={
                        "exit_code": proc.returncode,
                        "success": proc.returncode == 0,
                        "stdout_length": len(proc_result["stdout"]),
                        "stderr_length": len(proc_result["stderr"])
                    })
                    
            except subprocess.TimeoutExpired:
                proc.kill()
                out, err = proc.communicate()
                proc_result["stdout"] = out.decode("utf-8", errors="replace")
                proc_result["stderr"] = (
                    err.decode("utf-8", errors="replace")
                    + "\n[Execution terminated due to timeout]"
                )
                proc_result["exit_code"] = -1
                proc_result["success"] = False
                logger.warning("Sandbox execution timed out and was killed.")
                
                if notebook:
                    notebook.log("sandbox_runner", "EXECUTION_TIMEOUT", "Sandbox execution timed out and was killed", metadata={
                        "timeout": timeout,
                        "exit_code": -1,
                        "stdout_length": len(proc_result["stdout"]),
                        "stderr_length": len(proc_result["stderr"])
                    })
                    
        except Exception as e:
            logger.error(f"Sandboxed process failed: {e}")
            proc_result["stderr"] = str(e)
            proc_result["exit_code"] = -2
            proc_result["success"] = False
            
            if notebook:
                notebook.log("sandbox_runner", "EXECUTION_ERROR", f"Sandboxed process failed: {e}", metadata={
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "exit_code": -2
                })

    try:
        # Acquire a sandbox slot before starting execution
        sandbox_pool.acquire()
        
        thread_result: Dict[str, Any] = {}
        thread = threading.Thread(target=target, args=(thread_result,))
        thread.start()
        thread.join(timeout + 2)  # Give extra time to finish cleanup

        if not thread_result:
            error_msg = "Sandbox execution failed: no result captured"
            logger.error("Threaded sandbox execution failed: no result captured.")
            result["stderr"] = error_msg
            result["exit_code"] = -3
            result["success"] = False
            
            if notebook:
                notebook.log("sandbox_runner", "THREAD_ERROR", error_msg, metadata={
                    "exit_code": -3,
                    "thread_timeout": timeout + 2
                })
            
            return result

        result.update(thread_result)
        
        if notebook:
            notebook.log("sandbox_runner", "EXECUTION_COMPLETE", "Sandbox execution completed", metadata={
                "final_result": result,
                "success": result["success"],
                "exit_code": result["exit_code"]
            })
        
        return result
    
    finally:
        # Always release the sandbox slot
        sandbox_pool.release()


def run_python_file_in_sandbox(
    python_file_path: str,
    cpu_time_limit: int = 3,
    memory_limit_mb: int = 128,
    timeout: int = 5,
    extra_files: Optional[Dict[str, str]] = None,
    notebook: Optional[AddOnNotebook] = None,
) -> Dict[str, Any]:
    """
    Run a Python file in a secure, temporary sandbox directory with resource limits.

    Args:
        python_file_path: Path to the Python file to execute.
        cpu_time_limit: Max CPU time in seconds.
        memory_limit_mb: Max memory (MB).
        timeout: Wall-clock execution timeout (seconds).
        extra_files: Optional dict of {filename: contents} to place in sandbox.
        notebook: Optional AddOnNotebook instance for enhanced logging.

    Returns:
        Dict with keys: success (bool), stdout (str), stderr (str), exit_code (int).
    """
    result = {
        "success": False,
        "stdout": "",
        "stderr": "",
        "exit_code": None,
    }
    temp_dir = None

    if notebook:
        notebook.log("sandbox_runner", "SANDBOX_START", f"Starting sandbox execution for: {python_file_path}", metadata={
            "python_file_path": python_file_path,
            "cpu_time_limit": cpu_time_limit,
            "memory_limit_mb": memory_limit_mb,
            "timeout": timeout,
            "extra_files_count": len(extra_files) if extra_files else 0,
            "extra_files": list(extra_files.keys()) if extra_files else []
        })

    try:
        temp_dir = tempfile.mkdtemp(prefix="promethyn_sandbox_")
        logger.debug(f"Created sandbox temp dir at {temp_dir}")
        
        if notebook:
            notebook.log("sandbox_runner", "TEMP_DIR_CREATED", f"Created sandbox temp directory: {temp_dir}", metadata={
                "temp_dir": temp_dir
            })

        # Copy the Python file into the temp dir
        base_filename = os.path.basename(python_file_path)
        dest_file_path = safe_path_join(temp_dir, base_filename)
        shutil.copy2(python_file_path, dest_file_path)
        logger.debug(f"Copied {python_file_path} to sandbox as {dest_file_path}")
        
        if notebook:
            notebook.log("sandbox_runner", "FILE_COPIED", f"Copied Python file to sandbox", metadata={
                "source_path": python_file_path,
                "dest_path": dest_file_path,
                "base_filename": base_filename
            })

        # Write any extra files needed for the execution into the sandbox
        if extra_files:
            for fname, fcontent in extra_files.items():
                safe_path = safe_path_join(temp_dir, os.path.basename(fname))
                with open(safe_path, "w", encoding="utf-8") as f:
                    f.write(fcontent)
                logger.debug(f"Wrote extra file to sandbox: {safe_path}")
                
                if notebook:
                    notebook.log("sandbox_runner", "EXTRA_FILE_WRITTEN", f"Wrote extra file to sandbox: {fname}", metadata={
                        "filename": fname,
                        "safe_path": safe_path,
                        "content_length": len(fcontent)
                    })

        # Use a minimal, sanitized environment
        env = {
            "PATH": "/usr/bin:/bin",
            "LANG": "C.UTF-8",
            "PYTHONIOENCODING": "utf-8",
        }

        # Run the file with python in the sandbox dir
        cmd = [sys.executable, base_filename]
        logger.info(
            f"Executing {base_filename} in sandbox with CPU={cpu_time_limit}s, MEM={memory_limit_mb}MB, timeout={timeout}s"
        )

        if notebook:
            notebook.log("sandbox_runner", "COMMAND_PREPARED", f"Prepared sandbox command: {' '.join(cmd)}", metadata={
                "command": cmd,
                "base_filename": base_filename,
                "python_executable": sys.executable,
                "environment": env
            })

        exec_result = _run_with_limits(
            cmd,
            cwd=temp_dir,
            cpu_time_limit=cpu_time_limit,
            memory_limit_mb=memory_limit_mb,
            timeout=timeout,
            env=env,
            notebook=notebook,
        )
        result.update(exec_result)

        if notebook:
            notebook.log("sandbox_runner", "SANDBOX_EXECUTION_RESULT", f"Sandbox execution completed", metadata={
                "python_file_path": python_file_path,
                "success": result["success"],
                "exit_code": result["exit_code"],
                "stdout_length": len(result["stdout"]),
                "stderr_length": len(result["stderr"]),
                "full_result": result
            })

    except Exception as e:
        error_msg = f"Sandbox runner internal error: {e}"
        logger.exception(f"Sandbox failed: {e}")
        result["stderr"] = error_msg
        result["exit_code"] = -4
        result["success"] = False
        
        if notebook:
            notebook.log("sandbox_runner", "SANDBOX_INTERNAL_ERROR", f"Sandbox runner internal error: {e}", metadata={
                "python_file_path": python_file_path,
                "error": str(e),
                "error_type": type(e).__name__,
                "exit_code": -4,
                "temp_dir": temp_dir
            })
            
    finally:
        # Best effort to clean up
        if temp_dir and os.path.isdir(temp_dir):
            try:
                shutil.rmtree(temp_dir)
                logger.debug(f"Cleaned up sandbox temp dir {temp_dir}")
                
                if notebook:
                    notebook.log("sandbox_runner", "CLEANUP_SUCCESS", f"Successfully cleaned up sandbox temp directory", metadata={
                        "temp_dir": temp_dir
                    })
                    
            except Exception as e:
                logger.warning(f"Failed to clean up sandbox temp dir {temp_dir}: {e}")
                
                if notebook:
                    notebook.log("sandbox_runner", "CLEANUP_ERROR", f"Failed to clean up sandbox temp directory: {e}", metadata={
                        "temp_dir": temp_dir,
                        "error": str(e),
                        "error_type": type(e).__name__
                    })

    return result


# Example usage (for development, not production):
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run a Python file in a sandbox.")
    parser.add_argument("python_file", help="Path to the Python file to execute.")
    parser.add_argument(
        "--cpu", type=int, default=3, help="Max CPU time (seconds). Default: 3"
    )
    parser.add_argument(
        "--mem", type=int, default=128, help="Max memory (MB). Default: 128"
    )
    parser.add_argument(
        "--timeout", type=int, default=5, help="Max wall time (seconds). Default: 5"
    )

    args = parser.parse_args()

    # Create notebook instance for CLI usage
    notebook = AddOnNotebook()
    
    res = run_python_file_in_sandbox(
        args.python_file, 
        cpu_time_limit=args.cpu, 
        memory_limit_mb=args.mem, 
        timeout=args.timeout,
        notebook=notebook
    )
    print("Result:")
    print(res)
