import importlib
import os
import sys
from typing import Optional

REQUIRED_VERSION = dict(major=3, minor=11)

PACKAGE_IMPORT_REPLACEMENTS = {
    'pyyaml': 'yaml',
    'scikit-learn': 'sklearn',
    'huggingface-hub': 'huggingface_hub',
    'python-dotenv': 'dotenv'
}


def get_flag(name: str) -> bool:
    _val: Optional[str] = os.getenv(name)
    match _val:
        case "False" | "false" | "0" | None:
            return False
        case "True" | "true" | "1":
            return True
        case _:
            raise ValueError(f"Can't parse value of {name}=\"{_val}\"")


def verify() -> bool:
    # Check python version
    if sys.version_info.major != REQUIRED_VERSION['major']:
        raise RuntimeError(f"Expected major version {REQUIRED_VERSION['major']}, found {sys.version_info.major}")
    if sys.version_info.minor != REQUIRED_VERSION['minor']:
        raise RuntimeError(f"Expected minor version {REQUIRED_VERSION['minor']}, found {sys.version_info.minor}")
    print("> Python version correct")

    # Check packages
    assert os.path.exists("requirements.txt"), "requirements.txt not found, check working directory is project root"
    with open("requirements.txt") as f:
        reqs = f.readlines()
    reqs = [line.rstrip() for line in reqs]  # Clean newlines
    reqs = [line[:line.find("#")] for line in reqs]  # Clean comments
    reqs = [line[:line.find("--")] for line in reqs]  # Clean arguments to pip
    reqs = [line[:line.find("==")] for line in reqs]  # Clean version values
    requirements = [line for line in reqs if len(line) > 0]  # Clean newlines

    for module in requirements:
        if module in PACKAGE_IMPORT_REPLACEMENTS:
            module = PACKAGE_IMPORT_REPLACEMENTS[module]
        importlib.import_module(module)
        print(f"\t> {module} available")
    print("> All packages available")

    # Check .env file
    from dotenv import find_dotenv, load_dotenv
    load_dotenv(find_dotenv())  # This loads .env

    if not os.getenv("HELLO_FROM_ENV_FILE") == "ABCDEFGH":
        raise RuntimeError(".env missing, invalid or test value is set incorrectly")
    if os.getenv("PYTHONPATH") is None or os.getenv("PYTHONPATH") == "F:\\ULL\\PATH\\FROM\\DRIVE\\TO\\cs4nlp-plmrb":
        raise RuntimeError(".env has default PYTHONPATH, please update it")
    print("> .env usable")

    # Check for CUDA
    SKIP_CUDA_CHECK = get_flag("SKIP_CUDA_CHECK")

    if not SKIP_CUDA_CHECK:
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("Expected CUDA, but CUDA is not available")
        print(f"> CUDA v{torch.cuda_version} found")
    else:
        print("! Not checking for CUDA")

    # Finally
    print("> ALL CHECKS PASSED")
    return True


def main():
    verify()


if __name__ == '__main__':
    main()
