# Autohugg Library

The following section describes a part of the `Autohugg` library. This library is designed to simplify and automate certain tasks in your Python projects.

## Code Snippet from `main.py`

```python
    model_repo = HuggingFaceDatasetController(config,"dataset_name")
    model_repo.upload_dataset("./dir", DummyConfig(1), 1)


    c = HuggingFaceDatasetController(config,"dataset_name",repo_name ='autoencoder_data')
    c.download_latest_dataset('./dst_dir')
```



## Description

Interface to the Hugging Face Api
The module provides a simple interface to the Hugging Face Api. It allows to upload and download datasets and models to the Hugging Face Hub.
Its coupled to custom config types (dataclasses) that are used to configure the behavior of the desired operation.
It provides a DatasetContorler a torch model controller and and a StableBaselines controller.
Models and Datasets Branches are used to store different permutations/experiments of Datasets or Models.
A Dataset or Model is stored in a branch with a unique name. The branch is created if it does not exist.
Different Models or Datasets can be sorted in different repositories also.

## Usage

Here is a basic example of the DatasetController usage:


```python
from autohugg import HuggingFaceDatasetController# replace with actual module name

    model_repo = HuggingFaceDatasetController(config,"sub_dataset_name",repo_name ='dataset_name')
    model_repo.upload_dataset("./dir", DummyConfig(1), 1)


    c = HuggingFaceDatasetController(config,"sub_dataset_name",repo_name ='dataset_name')
    c.download_latest_dataset('./dst_dir')
```

Here is a basic example of the HuggingFaceAutoencoderModelController (torch) usage:


```python
from autohugg import HuggingFaceAutoencoderModelController# replace with actual module name

    model_repo = HuggingFaceAutoencoderModelController(config,"sub_model_name",repo_name ='dataset_name')
    model_repo.upload_dataset("./dir", DummyConfig(1), 1)


    c = HuggingFaceAutoencoderModelController(config,"sub_model_name",repo_name ='dataset_name')
    c.download_latest_dataset('./dst_dir')
```

Here is a basic example of the HuggingFaceAutoencoderStableBaseLinesController (sb3) usage:


```python
from autohugg import HuggingFaceStableBaseLinesModelController# replace with actual module name

    model_repo = HuggingFaceStableBaseLinesModelController(config,"sub_model_name",repo_name ='dataset_name')
    model_repo.upload_dataset("./dir", DummyConfig(1), 1)


    c = HuggingFaceStableBaseLinesModelController(config,"sub_model_name",repo_name ='dataset_name')
    c.download_latest_dataset('./dst_dir')
```
pass a config object to the controller to configure the behavior of the controller.
The config object is a dataclass that is used to configure the behavior of the controller.

the heracless package can be used to create a config object from a yaml file.

```python
from hugg.utils.load_config import load
config = load()
```
where the yaml file looks like this:
```yaml
title: Development Configuration

Hub:
  endpoint: "https://huggingface.co/<username>"
  username: "<username>"
  branch: "main"
  Model:
    repo_name: "test_model_0"
  Dataset:
    repo_name: "test_dataset_0"
```

## Installation

To install the `Autohugg` library, you can use the following command:

```bash 
git clone https://github.com/AdvancedAdditive/AutoHugg
pip install ./AutoHugg/
```

add the Huggingface Api Token as an environment variable
```bash
export READ_HUGG_TOKEN=<read_token>
export WRITE_HUGG_TOKEN=<write_token>
```
## Contributing

Contributions to the `Autohugg` library are welcome.

## License

The `Autohugg` library is licensed under the MIT license.
