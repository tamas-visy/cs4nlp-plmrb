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


def check() -> bool:
    # Check python version
    if sys.version_info.major != REQUIRED_VERSION['major']:
        raise EnvironmentError(f"Expected major version {REQUIRED_VERSION['major']}, found {sys.version_info.major}")
    if sys.version_info.minor != REQUIRED_VERSION['minor']:
        raise EnvironmentError(f"Expected minor version {REQUIRED_VERSION['minor']}, found {sys.version_info.minor}")
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

    # Check for CUDA
    from dotenv import find_dotenv, load_dotenv
    load_dotenv(find_dotenv())  # This loads .env

    _skip_cuda_check: Optional[str] = os.getenv("SKIP_CUDA_CHECK")
    match _skip_cuda_check:
        case "False" | "false" | "0" | None:
            SKIP_CUDA_CHECK = False
        case "True" | "true" | "1":
            SKIP_CUDA_CHECK = True
        case _:
            raise ValueError(f"Can't parse value of SKIP_CUDA_CHECK=\"{_skip_cuda_check}\"")

    if not SKIP_CUDA_CHECK:
        import torch
        if not torch.cuda.is_available():
            raise EnvironmentError("Expected CUDA, but CUDA is not available")
        print(f"> CUDA v{torch.cuda_version} found")
    else:
        print(f"! Not checking for CUDA")

    # Finally
    print("> ALL CHECKS PASSED")
    return True


def main():
    check()


if __name__ == '__main__':
    main()
