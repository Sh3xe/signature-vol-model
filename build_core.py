import subprocess
import sys
from pathlib import Path

def build():
    # Find all directories related to the engine
    root_dir = Path(__file__).parent.resolve()
    src_dir = root_dir / "signature_core"
    build_dir = src_dir / "build"

    # 1. Configure
    configure_cmd = [
        "cmake",
        "-S", str(src_dir),
        "-B", str(build_dir),
        "-DCMAKE_BUILD_TYPE=Release"
    ]
    
    # 2. Compile
    build_cmd = [
        "cmake",
        "--build", str(build_dir),
        "--config", "Release"
    ]

    try:
        print("Compiguring Cmake...")
        subprocess.run(configure_cmd, check=True)
        
        print("Compiling library...")
        subprocess.run(build_cmd, check=True)
        
        print("Successfully built the core library")
        
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Error while running the following command : {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    build()