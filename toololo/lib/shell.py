"""Shell command execution for toololo standard library."""

import os
import subprocess
import threading
import time
from pathlib import Path
from typing import Optional, Dict, Any, Callable, Union


class ShellCommandResult:
    """Result of a shell command execution."""
    
    def __init__(self, command: str, returncode: int, stdout: str, stderr: str, execution_time: float):
        self.command = command
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr
        self.execution_time = execution_time
    
    @property
    def success(self) -> bool:
        """Whether the command executed successfully (returncode == 0)."""
        return self.returncode == 0
    
    def __str__(self) -> str:
        if self.success:
            return self.stdout if self.stdout.strip() else "(no output)"
        else:
            error_msg = f"Command failed with exit code {self.returncode}"
            if self.stderr.strip():
                error_msg += f": {self.stderr}"
            elif self.stdout.strip():
                error_msg += f": {self.stdout}"
            return error_msg


def shell_command(
    command: str,
    working_directory: str = ".",
    timeout: int = 600,
    enable_environment: bool = False,
    streaming_callback: Optional[Callable[[str], None]] = None,
    **kwargs
) -> ShellCommandResult:
    """Execute an arbitrary shell command and return the output.
    
    Args:
        command: The shell command to execute
        working_directory: Directory to run the command in
        timeout: Timeout in seconds for command execution
        enable_environment: Whether to source shell initialization files
        streaming_callback: Optional callback for streaming output
        **kwargs: Additional keyword arguments (for compatibility)
        
    Returns:
        ShellCommandResult containing the command output and metadata
        
    Raises:
        FileNotFoundError: If working directory doesn't exist
        NotADirectoryError: If working directory path is not a directory  
        subprocess.TimeoutExpired: If command times out
        OSError: If command execution fails
    """
    if not command.strip():
        raise ValueError("Command cannot be empty")
    
    work_dir = Path(working_directory).expanduser().resolve()
    
    if not work_dir.exists():
        raise FileNotFoundError(f"Working directory does not exist: {working_directory}")
    
    if not work_dir.is_dir():
        raise NotADirectoryError(f"Working directory path is not a directory: {working_directory}")
    
    # Set up environment
    env = os.environ.copy()
    env['PAGER'] = 'cat'  # Prevent interactive pagers
    
    # Choose shell and arguments
    if enable_environment:
        # Interactive shell to source .bashrc and .bash_profile
        shell_cmd = ['bash', '-i', '-c', command]
    else:
        # Non-interactive shell
        shell_cmd = ['bash', '-c', command]
    
    start_time = time.time()
    
    try:
        if streaming_callback:
            # Use streaming execution
            result = _execute_with_streaming(
                shell_cmd, work_dir, timeout, env, streaming_callback
            )
        else:
            # Use standard execution
            result = subprocess.run(
                shell_cmd,
                cwd=work_dir,
                capture_output=True,
                text=True,
                timeout=timeout,
                env=env
            )
        
        execution_time = time.time() - start_time
        
        if hasattr(result, 'stdout'):
            # Standard subprocess.CompletedProcess
            return ShellCommandResult(
                command=command,
                returncode=result.returncode,
                stdout=result.stdout,
                stderr=result.stderr,
                execution_time=execution_time
            )
        else:
            # Custom streaming result
            return ShellCommandResult(
                command=command,
                returncode=result['returncode'],
                stdout=result['stdout'],
                stderr=result['stderr'],
                execution_time=execution_time
            )
    
    except subprocess.TimeoutExpired as e:
        execution_time = time.time() - start_time
        return ShellCommandResult(
            command=command,
            returncode=-1,
            stdout=e.stdout or "",
            stderr=f"Command timed out after {timeout} seconds",
            execution_time=execution_time
        )
    
    except Exception as e:
        execution_time = time.time() - start_time
        return ShellCommandResult(
            command=command,
            returncode=-1,
            stdout="",
            stderr=f"Failed to execute command: {str(e)}",
            execution_time=execution_time
        )


def _execute_with_streaming(
    shell_cmd: list, 
    work_dir: Path, 
    timeout: int, 
    env: Dict[str, str],
    streaming_callback: Callable[[str], None]
) -> Dict[str, Any]:
    """Execute command with streaming output."""
    
    process = subprocess.Popen(
        shell_cmd,
        cwd=work_dir,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
        bufsize=1,
        universal_newlines=True
    )
    
    stdout_lines = []
    stderr_lines = []
    
    def read_stdout():
        """Read stdout in a separate thread."""
        while True:
            line = process.stdout.readline()
            if not line:
                break
            stdout_lines.append(line.rstrip('\n'))
            streaming_callback(line.rstrip('\n'))
    
    def read_stderr():
        """Read stderr in a separate thread."""  
        while True:
            line = process.stderr.readline()
            if not line:
                break
            stderr_lines.append(line.rstrip('\n'))
    
    # Start reader threads
    stdout_thread = threading.Thread(target=read_stdout, daemon=True)
    stderr_thread = threading.Thread(target=read_stderr, daemon=True)
    
    stdout_thread.start()
    stderr_thread.start()
    
    # Wait for process to complete or timeout
    try:
        returncode = process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        process.kill()
        returncode = -1
        stderr_lines.append(f"Command timed out after {timeout} seconds")
    
    # Wait for threads to finish
    stdout_thread.join(timeout=1.0)
    stderr_thread.join(timeout=1.0)
    
    return {
        'returncode': returncode,
        'stdout': '\n'.join(stdout_lines),
        'stderr': '\n'.join(stderr_lines)
    }
