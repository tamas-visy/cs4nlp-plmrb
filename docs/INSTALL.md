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

Test environment with
```commandline
python test_environment.py
```

### torch and CUDA
Change the line "--extra-index-url https://download.pytorch.org/whl/cu118" in [requirements.txt](../requirements.txt)
to "--extra-index-url https://download.pytorch.org/whl/cpu" and update the version of `torch` similarly to install without CUDA support.

Additionally, to avoid an error when running `test_environment.py`, you should set the environment variable
`SKIP_CUDA_CHECK` to `True`. You can do this using the [.env](../.env) file or using the command line.