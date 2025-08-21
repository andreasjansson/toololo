"""Integration tests for toololo standard library."""

import os
import tempfile
import pytest
from pathlib import Path

from toololo.lib import files, shell


class TestFiles:
    """Test file operations."""
    
    def test_read_write_file(self):
        """Test reading and writing files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            test_file = Path(tmpdir) / "test.txt"
            content = "Hello, World!\nThis is a test file.\n"
            
            # Write file
            result = files.write_file(str(test_file), content)
            assert "Successfully wrote file" in result
            assert test_file.exists()
            
            # Read file
            read_content = files.read_file(str(test_file))
            assert read_content == content.rstrip('\n')  # read_file strips final newline
    
    def test_read_file_with_line_numbers(self):
        """Test reading file with line numbers."""
        with tempfile.TemporaryDirectory() as tmpdir:
            test_file = Path(tmpdir) / "numbered.txt"
            content = "Line 1\nLine 2\nLine 3\nLine 4\nLine 5"
            
            files.write_file(str(test_file), content)
            
            # Read with line numbers
            result = files.read_file(str(test_file), include_line_numbers=True)
            expected = "1: Line 1\n2: Line 2\n3: Line 3\n4: Line 4\n5: Line 5"
            assert result == expected
            
            # Read partial with line numbers
            result = files.read_file(str(test_file), include_line_numbers=True, start_line=2, end_line=4)
            expected = "2: Line 2\n3: Line 3\n4: Line 4"
            assert result == expected
    
    def test_str_replace(self):
        """Test string replacement in files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            test_file = Path(tmpdir) / "replace.txt"
            content = "Hello World\nThis is a test\nHello again"
            
            files.write_file(str(test_file), content)
            
            # Replace single occurrence
            result = files.str_replace(str(test_file), "Hello World", "Hi Universe")
            assert "Successfully replaced content" in result
            
            new_content = files.read_file(str(test_file))
            assert "Hi Universe" in new_content
            assert "Hello World" not in new_content
            assert "Hello again" in new_content  # Second occurrence unchanged
    
    def test_str_replace_all(self):
        """Test replacing all occurrences."""
        with tempfile.TemporaryDirectory() as tmpdir:
            test_file = Path(tmpdir) / "replace_all.txt"
            content = "foo bar foo baz foo"
            
            files.write_file(str(test_file), content)
            
            # Replace all occurrences
            result = files.str_replace(str(test_file), "foo", "FOO", replace_all=True)
            assert "made 3 replacements" in result
            
            new_content = files.read_file(str(test_file))
            assert new_content == "FOO bar FOO baz FOO"
    
    def test_str_replace_multiple_without_replace_all_fails(self):
        """Test that multiple occurrences fail without replace_all."""
        with tempfile.TemporaryDirectory() as tmpdir:
            test_file = Path(tmpdir) / "multi.txt"
            content = "foo bar foo baz"
            
            files.write_file(str(test_file), content)
            
            # Should fail because multiple occurrences found
            with pytest.raises(ValueError, match="Found 2 occurrences"):
                files.str_replace(str(test_file), "foo", "FOO")
    
    def test_delete_file(self):
        """Test deleting files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            test_file = Path(tmpdir) / "delete_me.txt"
            
            files.write_file(str(test_file), "content")
            assert test_file.exists()
            
            result = files.delete_file(str(test_file))
            assert "Successfully deleted file" in result
            assert not test_file.exists()
    
    def test_delete_files_multiple(self):
        """Test deleting multiple files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            files_to_delete = []
            for i in range(3):
                test_file = Path(tmpdir) / f"delete_{i}.txt"
                files.write_file(str(test_file), f"content {i}")
                files_to_delete.append(str(test_file))
            
            result = files.delete_files(files_to_delete)
            assert "Successfully deleted 3 file(s)" in result
            
            for file_path in files_to_delete:
                assert not Path(file_path).exists()
    
    def test_make_directory(self):
        """Test directory creation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            new_dir = Path(tmpdir) / "new" / "nested" / "directory"
            
            result = files.make_directory(str(new_dir))
            assert "Successfully created directory" in result
            assert new_dir.exists()
            assert new_dir.is_dir()
    
    def test_rename_file(self):
        """Test file renaming."""
        with tempfile.TemporaryDirectory() as tmpdir:
            old_path = Path(tmpdir) / "old_name.txt"
            new_path = Path(tmpdir) / "new_name.txt"
            
            files.write_file(str(old_path), "content")
            
            result = files.rename_file(str(old_path), str(new_path))
            assert "Successfully renamed" in result
            assert not old_path.exists()
            assert new_path.exists()
            
            content = files.read_file(str(new_path))
            assert content == "content"
    
    def test_list_directory(self):
        """Test directory listing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create some files and directories
            test_file1 = Path(tmpdir) / "file1.txt"
            test_file2 = Path(tmpdir) / "file2.txt"
            test_dir = Path(tmpdir) / "subdir"
            
            files.write_file(str(test_file1), "content1")
            files.write_file(str(test_file2), "content2")
            files.make_directory(str(test_dir))
            
            result = files.list_directory(str(tmpdir))
            assert "file1.txt" in result
            assert "file2.txt" in result
            assert "subdir" in result
            assert "(dir)" in result  # Directory marker
    
    def test_list_directory_recursive(self):
        """Test recursive directory listing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create nested structure
            subdir = Path(tmpdir) / "subdir"
            files.make_directory(str(subdir))
            
            files.write_file(str(Path(tmpdir) / "root.txt"), "root content")
            files.write_file(str(subdir / "nested.txt"), "nested content")
            
            result = files.list_directory(str(tmpdir), recursive=True)
            assert "root.txt" in result
            assert "subdir" in result
            assert "nested.txt" in result
            assert "./subdir/:" in result  # Nested directory header
    
    def test_file_errors(self):
        """Test error conditions."""
        # Test reading non-existent file
        with pytest.raises(FileNotFoundError):
            files.read_file("/nonexistent/path")
        
        # Test writing to existing file
        with tempfile.TemporaryDirectory() as tmpdir:
            test_file = Path(tmpdir) / "exists.txt"
            files.write_file(str(test_file), "content")
            
            with pytest.raises(FileExistsError):
                files.write_file(str(test_file), "new content")
        
        # Test replacing in non-existent file
        with pytest.raises(FileNotFoundError):
            files.str_replace("/nonexistent", "old", "new")
        
        # Test replacing non-existent content
        with tempfile.TemporaryDirectory() as tmpdir:
            test_file = Path(tmpdir) / "no_match.txt"
            files.write_file(str(test_file), "content")
            
            with pytest.raises(ValueError, match="Original content not found"):
                files.str_replace(str(test_file), "missing", "replacement")


class TestShell:
    """Test shell command execution."""
    
    def test_simple_command(self):
        """Test executing a simple command."""
        result = shell.shell_command("echo 'Hello World'")
        assert result.success
        assert result.returncode == 0
        assert "Hello World" in result.stdout
        assert result.execution_time > 0
    
    def test_command_with_working_directory(self):
        """Test command with specific working directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = shell.shell_command("pwd", working_directory=tmpdir)
            assert result.success
            # The output should contain the temp directory path
            assert str(Path(tmpdir).resolve()) in result.stdout
    
    def test_command_failure(self):
        """Test handling of failed commands."""
        result = shell.shell_command("exit 1")
        assert not result.success
        assert result.returncode == 1
    
    def test_command_with_output(self):
        """Test command that produces output."""
        result = shell.shell_command("echo 'line1'; echo 'line2'")
        assert result.success
        assert "line1" in result.stdout
        assert "line2" in result.stdout
    
    def test_command_timeout(self):
        """Test command timeout."""
        result = shell.shell_command("sleep 5", timeout=1)
        assert not result.success
        assert "timed out" in result.stderr.lower()
    
    def test_streaming_callback(self):
        """Test streaming output callback."""
        captured_output = []
        
        def callback(output):
            captured_output.append(output)
        
        result = shell.shell_command(
            "echo 'first'; sleep 0.1; echo 'second'",
            streaming_callback=callback
        )
        
        assert result.success
        assert len(captured_output) >= 2
        assert any("first" in line for line in captured_output)
        assert any("second" in line for line in captured_output)
    
    def test_environment_variables(self):
        """Test environment variable handling."""
        result = shell.shell_command("echo $PAGER")
        assert result.success
        assert "cat" in result.stdout  # PAGER should be set to cat
    
    def test_enable_environment(self):
        """Test enable_environment parameter."""
        # This test may behave differently depending on shell config
        # Just ensure it doesn't crash
        result = shell.shell_command("echo $HOME", enable_environment=True)
        assert result.success  # Should complete successfully
    
    def test_multiline_output(self):
        """Test commands with multiline output."""
        result = shell.shell_command("printf 'line1\\nline2\\nline3'")
        assert result.success
        lines = result.stdout.split('\n')
        assert "line1" in lines
        assert "line2" in lines
        assert "line3" in lines
    
    def test_command_with_special_characters(self):
        """Test commands with special characters."""
        result = shell.shell_command("echo 'Hello & goodbye; echo nested | grep test'")
        assert result.success
        assert "Hello & goodbye" in result.stdout
    
    def test_shell_command_result_str(self):
        """Test ShellCommandResult string representation."""
        # Successful command
        result = shell.shell_command("echo 'test output'")
        assert "test output" in str(result)
        
        # Failed command
        result = shell.shell_command("nonexistent_command_xyz")
        result_str = str(result)
        assert "Command failed" in result_str or "not found" in result_str.lower()
    
    def test_empty_command_error(self):
        """Test that empty commands raise ValueError."""
        with pytest.raises(ValueError, match="Command cannot be empty"):
            shell.shell_command("")
        
        with pytest.raises(ValueError, match="Command cannot be empty"):
            shell.shell_command("   ")
    
    def test_invalid_working_directory(self):
        """Test error handling for invalid working directory."""
        with pytest.raises(FileNotFoundError):
            shell.shell_command("echo test", working_directory="/nonexistent/directory")
        
        # Test with a file instead of directory
        with tempfile.NamedTemporaryFile() as tmp:
            with pytest.raises(NotADirectoryError):
                shell.shell_command("echo test", working_directory=tmp.name)


class TestIntegration:
    """Integration tests combining files and shell operations."""
    
    def test_create_and_execute_script(self):
        """Test creating a script file and executing it."""
        with tempfile.TemporaryDirectory() as tmpdir:
            script_path = Path(tmpdir) / "test_script.sh"
            script_content = """#!/bin/bash
echo "Script is running"
echo "Current directory: $(pwd)"
echo "Arguments: $@"
"""
            
            # Create script file
            files.write_file(str(script_path), script_content)
            
            # Make script executable and run it
            result = shell.shell_command(f"chmod +x {script_path} && {script_path} arg1 arg2")
            assert result.success
            assert "Script is running" in result.stdout
            assert "Arguments: arg1 arg2" in result.stdout
    
    def test_process_file_with_commands(self):
        """Test processing a file using shell commands."""
        with tempfile.TemporaryDirectory() as tmpdir:
            data_file = Path(tmpdir) / "data.txt"
            processed_file = Path(tmpdir) / "processed.txt"
            
            # Create data file
            data_content = "apple\nbanana\ncherry\ndate\napricot"
            files.write_file(str(data_file), data_content)
            
            # Process file with shell commands (sort and filter)
            result = shell.shell_command(
                f"sort {data_file} | grep ^a > {processed_file}",
                working_directory=tmpdir
            )
            assert result.success
            
            # Read processed file
            processed_content = files.read_file(str(processed_file))
            lines = processed_content.split('\n')
            assert "apple" in lines
            assert "apricot" in lines
            assert "banana" not in processed_content
    
    def test_file_modification_workflow(self):
        """Test a complete file modification workflow."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_file = Path(tmpdir) / "config.txt"
            
            # Create initial config
            initial_content = """# Configuration file
server_name=localhost
port=8080
debug=false
"""
            files.write_file(str(config_file), initial_content)
            
            # Modify config using str_replace
            files.str_replace(str(config_file), "port=8080", "port=9090")
            files.str_replace(str(config_file), "debug=false", "debug=true")
            
            # Verify changes with shell command
            result = shell.shell_command(f"grep port {config_file}")
            assert result.success
            assert "port=9090" in result.stdout
            
            result = shell.shell_command(f"grep debug {config_file}")
            assert result.success
            assert "debug=true" in result.stdout
            
            # Create backup
            backup_file = str(config_file) + ".bak"
            files.rename_file(str(config_file), backup_file)
            
            # Verify backup exists and original is gone
            assert Path(backup_file).exists()
            assert not config_file.exists()
    
    def test_directory_operations_with_shell(self):
        """Test directory operations combined with shell commands."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create directory structure
            project_dir = Path(tmpdir) / "project"
            src_dir = project_dir / "src"  
            docs_dir = project_dir / "docs"
            
            files.make_directory(str(src_dir))
            files.make_directory(str(docs_dir))
            
            # Create some files
            files.write_file(str(src_dir / "main.py"), "print('Hello')")
            files.write_file(str(src_dir / "utils.py"), "def helper(): pass")
            files.write_file(str(docs_dir / "README.md"), "# Project Docs")
            
            # Use shell to count files
            result = shell.shell_command("find . -name '*.py' | wc -l", working_directory=str(project_dir))
            assert result.success
            assert "2" in result.stdout.strip()  # Should find 2 Python files
            
            # List directory structure
            listing = files.list_directory(str(project_dir), recursive=True)
            assert "main.py" in listing
            assert "utils.py" in listing
            assert "README.md" in listing
