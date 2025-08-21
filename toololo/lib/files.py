"""File operations for toololo standard library."""

import os
import shutil
from pathlib import Path
from typing import Optional, List


def read_file(
    path: str,
    include_line_numbers: bool = False,
    start_line: Optional[int] = None,
    end_line: Optional[int] = None
) -> str:
    """Read the contents of a file from the filesystem.
    
    Args:
        path: Path to the file to read
        include_line_numbers: Whether to include line numbers in the output
        start_line: Starting line number (1-based) to begin reading from
        end_line: Ending line number (1-based) to stop reading at (inclusive)
        
    Returns:
        File contents as string
        
    Raises:
        FileNotFoundError: If file doesn't exist
        PermissionError: If file isn't readable
        IsADirectoryError: If path is a directory
        ValueError: If line numbers are invalid
    """
    file_path = Path(path).expanduser().resolve()
    
    if not file_path.exists():
        raise FileNotFoundError(f"File does not exist: {path}")
    
    if not file_path.is_file():
        raise IsADirectoryError(f"Path is a directory, not a file: {path}")
    
    if not os.access(file_path, os.R_OK):
        raise PermissionError(f"File is not readable: {path}")
    
    if start_line is not None and start_line < 1:
        raise ValueError("start_line must be >= 1")
    
    if end_line is not None and end_line < 1:
        raise ValueError("end_line must be >= 1")
    
    if start_line is not None and end_line is not None and start_line > end_line:
        raise ValueError("start_line must be <= end_line")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.read().splitlines()
    
    if start_line is not None:
        start_idx = start_line - 1
    else:
        start_idx = 0
        
    if end_line is not None:
        end_idx = end_line
    else:
        end_idx = len(lines)
    
    selected_lines = lines[start_idx:end_idx]
    
    if include_line_numbers:
        line_num = (start_line or 1)
        max_width = len(str(line_num + len(selected_lines) - 1))
        numbered_lines = []
        
        for line in selected_lines:
            numbered_lines.append(f"{line_num:{max_width}d}: {line}")
            line_num += 1
        
        return '\n'.join(numbered_lines)
    else:
        return '\n'.join(selected_lines)


def write_file(path: str, contents: str) -> str:
    """Write contents to a new file. Fails if file already exists.
    
    Args:
        path: Absolute path to the new file
        contents: Contents to write to the new file
        
    Returns:
        Success message
        
    Raises:
        FileExistsError: If file already exists
        OSError: If write fails
    """
    file_path = Path(path).expanduser().resolve()
    
    if file_path.exists():
        raise FileExistsError(f"File already exists: {path}")
    
    # Create parent directories if they don't exist
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(contents)
    
    return f"Successfully wrote file {file_path} with {len(contents)} characters"


def str_replace(
    path: str,
    original_content: str,
    new_content: str,
    replace_all: bool = False
) -> str:
    """Replace specific string or content block in a file with new content.
    
    Args:
        path: Path to the file to modify
        original_content: The exact content to find and replace
        new_content: The new content to replace the original content with
        replace_all: If True, replace all instances; if False, replace only first
        
    Returns:
        Success message
        
    Raises:
        FileNotFoundError: If file doesn't exist
        IsADirectoryError: If path is a directory
        ValueError: If original content not found or multiple found when replace_all=False
    """
    file_path = Path(path).expanduser().resolve()
    
    if not file_path.exists():
        raise FileNotFoundError(f"File does not exist: {path}")
    
    if not file_path.is_file():
        raise IsADirectoryError(f"Path is a directory, not a file: {path}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        contents = f.read()
    
    # Count occurrences
    occurrence_count = contents.count(original_content)
    
    if occurrence_count == 0:
        raise ValueError(f"Original content not found in file: {path}")
    
    if not replace_all and occurrence_count > 1:
        raise ValueError(
            f"Found {occurrence_count} occurrences of original content in file: {path}. "
            f"Use replace_all=True to replace all instances, or make the original content more specific"
        )
    
    # Perform replacement
    if replace_all:
        new_contents = contents.replace(original_content, new_content)
        replacements_made = occurrence_count
    else:
        new_contents = contents.replace(original_content, new_content, 1)
        replacements_made = 1
    
    # Write back to file
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(new_contents)
    
    count_msg = f" (made {replacements_made} replacements)" if replacements_made > 1 else ""
    return f"Successfully replaced content in {file_path}{count_msg}"


def delete_file(path: str) -> str:
    """Delete a file.
    
    Args:
        path: Path to the file to delete
        
    Returns:
        Success message
        
    Raises:
        FileNotFoundError: If file doesn't exist
        IsADirectoryError: If path is a directory
    """
    file_path = Path(path).expanduser().resolve()
    
    if not file_path.exists():
        raise FileNotFoundError(f"File does not exist: {path}")
    
    if not file_path.is_file():
        raise IsADirectoryError(f"Cannot delete directories: {path} (only files are supported)")
    
    file_path.unlink()
    return f"Successfully deleted file: {file_path}"


def delete_files(paths: List[str]) -> str:
    """Delete multiple files.
    
    Args:
        paths: List of file paths to delete
        
    Returns:
        Success message
        
    Raises:
        FileNotFoundError: If any file doesn't exist
        IsADirectoryError: If any path is a directory
    """
    resolved_paths = []
    
    # Validate all files exist first
    for path in paths:
        file_path = Path(path).expanduser().resolve()
        if not file_path.exists():
            raise FileNotFoundError(f"File does not exist: {path}")
        if not file_path.is_file():
            raise IsADirectoryError(f"Cannot delete directories: {path} (only files are supported)")
        resolved_paths.append(file_path)
    
    # Delete all files
    for file_path in resolved_paths:
        file_path.unlink()
    
    deleted_list = ", ".join(str(p) for p in resolved_paths)
    return f"Successfully deleted {len(resolved_paths)} file(s): {deleted_list}"


def make_directory(path: str) -> str:
    """Recursively create a directory and all parent directories if they don't exist.
    
    Args:
        path: Path to the directory to create
        
    Returns:
        Success message
    """
    dir_path = Path(path).expanduser().resolve()
    
    if dir_path.exists():
        if dir_path.is_dir():
            return f"Directory already exists: {dir_path}"
        else:
            raise FileExistsError(f"Path exists but is not a directory: {dir_path}")
    
    dir_path.mkdir(parents=True, exist_ok=True)
    return f"Successfully created directory: {dir_path}"


def rename_file(old_path: str, new_path: str) -> str:
    """Rename or move a file from one path to another.
    
    Args:
        old_path: Current path of the file
        new_path: New path for the file
        
    Returns:
        Success message
        
    Raises:
        FileNotFoundError: If source file doesn't exist
        FileExistsError: If destination file already exists
    """
    old_file_path = Path(old_path).expanduser().resolve()
    new_file_path = Path(new_path).expanduser().resolve()
    
    if not old_file_path.exists():
        raise FileNotFoundError(f"Source file does not exist: {old_path}")
    
    if new_file_path.exists():
        raise FileExistsError(f"Destination file already exists: {new_path}")
    
    # Ensure destination directory exists
    new_file_path.parent.mkdir(parents=True, exist_ok=True)
    
    old_file_path.rename(new_file_path)
    return f"Successfully renamed {old_file_path} to {new_file_path}"


def list_directory(
    path: str = ".",
    exclude_directories_recursive: Optional[List[str]] = None,
    recursive: bool = False
) -> str:
    """List files and directories in a given directory.
    
    Args:
        path: Path to the directory to list. Defaults to current directory.
        exclude_directories_recursive: List of directory names to exclude when recursively listing files
        recursive: Whether to list files recursively
        
    Returns:
        Directory listing as string
        
    Raises:
        FileNotFoundError: If directory doesn't exist
        NotADirectoryError: If path is not a directory
        PermissionError: If directory is not readable
    """
    if exclude_directories_recursive is None:
        exclude_directories_recursive = [".git", "__pycache__"]
    
    dir_path = Path(path).expanduser().resolve()
    
    if not dir_path.exists():
        raise FileNotFoundError(f"Directory does not exist: {path}")
    
    if not dir_path.is_dir():
        raise NotADirectoryError(f"Path is not a directory: {path}")
    
    if not os.access(dir_path, os.R_OK):
        raise PermissionError(f"Directory is not readable: {path}")
    
    def _list_directory_recursive(current_path: Path, prefix: str = "") -> List[str]:
        """Recursively list directory contents."""
        results = []
        
        # Display path
        display_path = str(current_path) if prefix == "" else f"./{prefix}"
        results.append(f"{display_path}:")
        
        try:
            # Get directory contents
            entries = sorted(current_path.iterdir(), key=lambda p: (p.is_file(), p.name.lower()))
            
            # Add . and .. entries
            results.append(_format_file_info(current_path, "."))
            if current_path != current_path.parent:
                results.append(_format_file_info(current_path.parent, ".."))
            
            # Process files and directories
            subdirs = []
            for entry in entries:
                if entry.name.startswith('.') and entry.name not in ['.', '..']:
                    continue  # Skip hidden files/dirs except . and ..
                
                results.append(_format_file_info(entry, entry.name))
                
                # Collect subdirectories for recursive processing
                if (recursive and entry.is_dir() and 
                    entry.name not in exclude_directories_recursive):
                    subdirs.append(entry)
            
            # Process subdirectories recursively
            if recursive:
                for subdir in subdirs:
                    subdir_prefix = f"{prefix}{subdir.name}/" if prefix else f"{subdir.name}/"
                    subdir_results = _list_directory_recursive(subdir, subdir_prefix)
                    results.extend([""] + subdir_results)
                    
        except PermissionError:
            results.append("Permission denied")
        
        return results
    
    def _format_file_info(file_path: Path, display_name: str) -> str:
        """Format file information similar to ls -la output."""
        try:
            stat = file_path.stat()
            
            # File type and permissions
            if file_path.is_dir():
                mode_str = "drwxrwxrwx"
                size_str = "(dir)"
            elif file_path.is_symlink():
                mode_str = "lrwxrwxrwx"
                size_str = f"{stat.st_size:8d}"
            else:
                mode_str = "-rwxrwxrwx"
                size_str = f"{stat.st_size:8d}"
            
            return f"{mode_str}  {size_str}  {display_name}"
            
        except (OSError, PermissionError):
            return f"?---------        ?  {display_name}"
    
    results = _list_directory_recursive(dir_path)
    
    if len(results) <= 1:  # Only header
        return f"{dir_path}/:\nDirectory is empty"
    
    return "\n".join(results)
