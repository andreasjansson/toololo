#!/usr/bin/env python3
"""
Example script demonstrating the toololo standard library.

This script shows how to use the toololo.lib modules for file operations
and shell command execution without git commits.
"""

import tempfile
from pathlib import Path

from toololo.lib import files, shell


def main():
    """Main example demonstrating toololo.lib usage."""
    
    print("🔧 toololo.lib Example")
    print("=" * 50)
    
    # Create a temporary directory for our demo
    with tempfile.TemporaryDirectory() as tmpdir:
        print(f"Working in temporary directory: {tmpdir}")
        
        # === File Operations Demo ===
        print("\n📁 File Operations:")
        
        # Create a sample file
        sample_file = Path(tmpdir) / "sample.txt"
        content = """Hello, toololo.lib!
This is a demonstration of file operations.
We can read, write, and modify files easily.
Line 4: Numbers and symbols: 123 !@# $%^
Line 5: Unicode: 🚀 ✨ 🎉"""
        
        result = files.write_file(str(sample_file), content)
        print(f"✅ {result}")
        
        # Read the file back
        read_content = files.read_file(str(sample_file))
        print(f"📖 File content:\n{read_content}\n")
        
        # Read with line numbers
        numbered_content = files.read_file(str(sample_file), include_line_numbers=True, start_line=2, end_line=4)
        print(f"📖 Lines 2-4 with numbers:\n{numbered_content}\n")
        
        # Modify file with string replacement
        result = files.str_replace(str(sample_file), "demonstration", "DEMO")
        print(f"✏️ {result}")
        
        # === Directory Operations Demo ===
        print("\n📂 Directory Operations:")
        
        # Create nested directories
        project_dir = Path(tmpdir) / "my_project" / "src"
        result = files.make_directory(str(project_dir))
        print(f"📁 {result}")
        
        # Create multiple files
        for i, filename in enumerate(["main.py", "utils.py", "config.json"], 1):
            file_path = project_dir / filename
            file_content = f"# File {i}: {filename}\nprint('Hello from {filename}')"
            files.write_file(str(file_path), file_content)
        
        # List directory structure
        print(f"📋 Directory listing:")
        listing = files.list_directory(str(Path(tmpdir) / "my_project"), recursive=True)
        print(listing)
        
        # === Shell Commands Demo ===
        print("\n💻 Shell Commands:")
        
        # Simple command
        result = shell.shell_command("echo 'Hello from shell!'")
        print(f"📤 Command output: {result.stdout}")
        print(f"⏱️ Execution time: {result.execution_time:.3f}s")
        
        # Command with working directory
        result = shell.shell_command("ls -la", working_directory=str(project_dir))
        print(f"📂 Files in project directory:")
        print(result.stdout)
        
        # Command with multiple steps
        result = shell.shell_command(
            "find . -name '*.py' | wc -l",
            working_directory=str(Path(tmpdir) / "my_project")
        )
        print(f"🔍 Python files found: {result.stdout.strip()}")
        
        # Command with timeout demonstration
        print("\n⏰ Timeout demonstration (quick command):")
        result = shell.shell_command("sleep 0.1 && echo 'Done!'", timeout=2)
        if result.success:
            print(f"✅ Completed: {result.stdout}")
        else:
            print(f"❌ Failed: {result.stderr}")
        
        # === Integration Example ===
        print("\n🔄 Integration Example:")
        
        # Create a data processing pipeline
        data_file = Path(tmpdir) / "numbers.txt"
        numbers = "\n".join(str(i) for i in [5, 2, 8, 1, 9, 3, 7, 4, 6])
        files.write_file(str(data_file), numbers)
        print(f"📊 Created data file with numbers")
        
        # Process with shell command
        result = shell.shell_command(f"sort -n {data_file}")
        if result.success:
            sorted_numbers = result.stdout.strip().replace('\n', ', ')
            print(f"🔢 Sorted numbers: {sorted_numbers}")
        
        # Save sorted result to new file
        sorted_file = Path(tmpdir) / "sorted_numbers.txt"
        files.write_file(str(sorted_file), result.stdout)
        print(f"💾 Saved sorted numbers to {sorted_file.name}")
        
        # === Error Handling Demo ===
        print("\n⚠️ Error Handling:")
        
        # Demonstrate error handling
        try:
            files.read_file("/nonexistent/file.txt")
        except FileNotFoundError as e:
            print(f"🚨 Caught expected error: {e}")
        
        try:
            result = shell.shell_command("nonexistent_command_xyz")
            if not result.success:
                print(f"🚨 Command failed as expected: {result}")
        except Exception as e:
            print(f"🚨 Shell error: {e}")
        
        # === Streaming Demo ===
        print("\n📡 Streaming Output Demo:")
        
        captured_lines = []
        
        def stream_callback(line):
            print(f"📺 Stream: {line}")
            captured_lines.append(line)
        
        result = shell.shell_command(
            "for i in {1..3}; do echo 'Line $i'; sleep 0.1; done",
            streaming_callback=stream_callback
        )
        
        print(f"🎬 Streaming captured {len(captured_lines)} lines")
        
        print("\n✨ Demo completed successfully!")


if __name__ == "__main__":
    main()
