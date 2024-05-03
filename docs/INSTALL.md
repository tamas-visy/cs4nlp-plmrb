# Installation and setup

## Basic requirements

Python interpreter, conda. This repository.

## Environment

```commandline
conda create --name cs4nlp-plmrb python=3.11
```

```commandline
conda activate cs4nlp-plmrb
```

```commandline
pip install -r requirements.txt
```

### torch and CUDA
Change the line "--extra-index-url https://download.pytorch.org/whl/cu118" in [requirements.txt](../requirements.txt)
to "--extra-index-url https://download.pytorch.org/whl/cpu" and update the version of `torch` similarly to install without CUDA support.

Additionally, to avoid an error when running `test_environment.py`, you should set the environment variable
`SKIP_CUDA_CHECK` to `True`. You can do this using the [.env](../.env) file or using the command line.

## Config using .env

Create a [.env](../.env) file in the repository root like this:
```
# Environment variables go here, can be read by `python-dotenv` package:
#
#   `src/script.py`
#   ----------------------------------------------------------------
#    import dotenv
#
#    project_dir = os.path.join(os.path.dirname(__file__), os.pardir)
#    dotenv_path = os.path.join(project_dir, '.env')
#    dotenv.load_dotenv(dotenv_path)
#   ----------------------------------------------------------------
#
# DO NOT ADD THIS FILE TO VERSION CONTROL!
HELLO_FROM_ENV_FILE = ABCDEFGH
#   ----------------------------------------------------------------

```

After the last line, feel free to add any settings you want, such as `LOGGING_LEVEL = DEBUG` or `SKIP_CUDA_CHECK = True`.

## Test environment
Test environment with
```commandline
python test_environment.py
```