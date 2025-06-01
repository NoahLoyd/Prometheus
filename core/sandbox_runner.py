import os
import tempfile
import shutil
import subprocess
import sys
import logging
import resource
import threading
from typing import Dict, Any, Optional

# Promethyn logger setup — assumes logging config elsewhere
logger = logging.getLogger("promethyn.sandbox")


class SandboxTimeout(Exception):
    pass


class SandboxPool:
    """
    Controls concurrent access to sandbox resources using a semaphore.
    """
    def __init__(self, max_concurrent: int):
        self.semaphore = threading.Semaphore(max_concurrent)
        logger.info(f"Initialized SandboxPool with {max_concurrent} concurrent slots")
    
    def acquire(self):
        """Acquire a sandbox slot"""
        acquired = self.semaphore.acquire(blocking=True)
        if acquired:
            logger.debug("Acquired sandbox slot")
        return acquired
    
    def release(self):
        """Release a sandbox slot"""
        self.semaphore.release()
        logger.debug("Released sandbox slot")


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
        except Exception as e:
            logger.error(f"Sandboxed process failed: {e}")
            proc_result["stderr"] = str(e)
            proc_result["exit_code"] = -2
            proc_result["success"] = False

    try:
        # Acquire a sandbox slot before starting execution
        sandbox_pool.acquire()
        
        thread_result: Dict[str, Any] = {}
        thread = threading.Thread(target=target, args=(thread_result,))
        thread.start()
        thread.join(timeout + 2)  # Give extra time to finish cleanup

        if not thread_result:
            logger.error("Threaded sandbox execution failed: no result captured.")
            result["stderr"] = "Sandbox execution failed: no result captured"
            result["exit_code"] = -3
            result["success"] = False
            return result

        result.update(thread_result)
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
) -> Dict[str, Any]:
    """
    Run a Python file in a secure, temporary sandbox directory with resource limits.

    Args:
        python_file_path: Path to the Python file to execute.
        cpu_time_limit: Max CPU time in seconds.
        memory_limit_mb: Max memory (MB).
        timeout: Wall-clock execution timeout (seconds).
        extra_files: Optional dict of {filename: contents} to place in sandbox.

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

    try:
        temp_dir = tempfile.mkdtemp(prefix="promethyn_sandbox_")
        logger.debug(f"Created sandbox temp dir at {temp_dir}")

        # Copy the Python file into the temp dir
        base_filename = os.path.basename(python_file_path)
        dest_file_path = os.path.join(temp_dir, base_filename)
        shutil.copy2(python_file_path, dest_file_path)
        logger.debug(f"Copied {python_file_path} to sandbox as {dest_file_path}")

        # Write any extra files needed for the execution into the sandbox
        if extra_files:
            for fname, fcontent in extra_files.items():
                safe_path = os.path.join(temp_dir, os.path.basename(fname))
                with open(safe_path, "w", encoding="utf-8") as f:
                    f.write(fcontent)
                logger.debug(f"Wrote extra file to sandbox: {safe_path}")

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

        exec_result = _run_with_limits(
            cmd,
            cwd=temp_dir,
            cpu_time_limit=cpu_time_limit,
            memory_limit_mb=memory_limit_mb,
            timeout=timeout,
            env=env,
        )
        result.update(exec_result)

    except Exception as e:
        logger.exception(f"Sandbox failed: {e}")
        result["stderr"] = f"Sandbox runner internal error: {e}"
        result["exit_code"] = -4
        result["success"] = False
    finally:
        # Best effort to clean up
        if temp_dir and os.path.isdir(temp_dir):
            try:
                shutil.rmtree(temp_dir)
                logger.debug(f"Cleaned up sandbox temp dir {temp_dir}")
            except Exception as e:
                logger.warning(f"Failed to clean up sandbox temp dir {temp_dir}: {e}")

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

    res = run_python_file_in_sandbox(
        args.python_file, cpu_time_limit=args.cpu, memory_limit_mb=args.mem, timeout=args.timeout
    )
    print("Result:")
    print(res)
