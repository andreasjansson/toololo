#!/usr/bin/env python3

import logging
import sys
import os

# Set up comprehensive logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

# Enable debug logging for our modules specifically
logging.getLogger('toololo').setLevel(logging.DEBUG)
logging.getLogger('toololo.run').setLevel(logging.DEBUG)
logging.getLogger('toololo.function').setLevel(logging.DEBUG)

# Also enable debug for openai if we want to see API details
logging.getLogger('openai').setLevel(logging.INFO)

print("=" * 80)
print("RUNNING INTEGRATION TESTS WITH VERBOSE LOGGING")
print("=" * 80)

if __name__ == "__main__":
    # Now run the tests using subprocess so we can see all the output
    import subprocess
    
    cmd = [
        sys.executable, "-m", "pytest", 
        "-vv", "-s", "--tb=short", 
        "test-integration/"
    ]
    
    print(f"Running command: {' '.join(cmd)}")
    print("=" * 80)
    
    result = subprocess.run(cmd, env={**os.environ})
    sys.exit(result.returncode)
