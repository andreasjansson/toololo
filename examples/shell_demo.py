#!/usr/bin/env python3
"""
Demo script showing different ways to use the shell library.
"""

import tempfile
from pathlib import Path

# Different import styles
from toololo.lib import shell_command  # Direct function import
from toololo.lib import shell          # Module import
import toololo.lib                     # Package import


def main():
    print("🐚 Shell Library Demo")
    print("=" * 40)
    
    # Method 1: Direct function import
    print("\n1️⃣ Direct function import:")
    result = shell_command("echo 'Hello from direct import'")
    print(f"   Output: {result.stdout}")
    print(f"   Success: {result.success}")
    
    # Method 2: Module import
    print("\n2️⃣ Module import:")
    result = shell.shell_command("echo 'Hello from module import'")
    print(f"   Output: {result.stdout}")
    print(f"   Return code: {result.returncode}")
    
    # Method 3: Package import  
    print("\n3️⃣ Package import:")
    result = toololo.lib.shell_command("echo 'Hello from package import'")
    print(f"   Output: {result.stdout}")
    print(f"   Execution time: {result.execution_time:.3f}s")
    
    # Advanced features
    print("\n🚀 Advanced Features:")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Working directory
        result = shell_command("pwd", working_directory=tmpdir)
        print(f"   Working dir test: {Path(result.stdout.strip()).name}")
        
        # Timeout
        result = shell_command("sleep 0.1", timeout=5)
        print(f"   Timeout test: {'✅ Success' if result.success else '❌ Failed'}")
        
        # Error handling
        result = shell_command("exit 42")
        print(f"   Error handling: Code {result.returncode}, Success: {result.success}")
        
        # Streaming
        print("   Streaming test:")
        lines = []
        def capture(line):
            lines.append(line)
            print(f"     📡 {line}")
        
        result = shell_command(
            "echo 'Line 1'; echo 'Line 2'; echo 'Line 3'",
            streaming_callback=capture
        )
        print(f"     Captured {len(lines)} lines via streaming")
    
    print("\n✨ All shell library methods working!")


if __name__ == "__main__":
    main()
