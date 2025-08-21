"""toololo.lib - Standard library functions for toololo."""

from . import files
from . import shell

# Re-export main functions for convenience
from .files import (
    read_file, write_file, str_replace, delete_file, delete_files,
    make_directory, rename_file, list_directory
)
from .shell import shell_command, ShellCommandResult

__all__ = [
    'files', 'shell',
    # File functions
    'read_file', 'write_file', 'str_replace', 'delete_file', 'delete_files',
    'make_directory', 'rename_file', 'list_directory',
    # Shell functions
    'shell_command', 'ShellCommandResult'
]
