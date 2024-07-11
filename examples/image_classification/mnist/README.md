# MNIST Compressing Example
<!--TOC-->

- [Requirements](#requirements)
- [Prepare Dataset](#prepare-dataset)
- [Run Example](#run-example)
- [Deploy `.pompom` file](#deploy-pompom-file)
- [References](#references)

<!--TOC-->

**CLIKA SDK** example to compress CNN model on `MNIST` dataset

## Requirements

- Python >= 3.8
- CLIKA SDK (<https://docs.clika.io/docs/installation>)
- `pip install torchmetrics==1.3.2`

## Prepare Dataset

Dataset will be automatically downloaded using PyTorch into `image_classification/mnist/MNIST`
when running `mnist_main.py` the first time

After running the `mnist_main.py` script the first time,
the dataset directory tree should look like the following:

```text
image_classification/mnist/MNIST/
├── raw/
│   ├── t10k-images-idx3-ubyte
│   ├── t10k-images-idx3-ubyte.gz
│   ├── t10k-images-idx1-ubyte
│   ├── t10k-images-idx1-ubyte.gz
...

```

## Run Example

```shell
# pwd: image_classification/mnist
python3 mnist_main.py --output_dir outputs
```

## Deploy `.pompom` file

```python
from clika_compression.utils import get_path_to_best_clika_state_result
from clika_compression import clika_deploy


# (OPTIONAL) find the best performing pompom file
best_pompom_file_path: str = get_path_to_best_clika_state_result(
    "outputs",
    key_name="acc",
    summary_json_group="evaluation",
    find_lowest=False,
)

deployed_model_path: str = clika_deploy(
    clika_state_path=best_pompom_file_path,
    output_dir_path="outputs",

    # (OPTIONAL)set this if you wish to use dynamic shapes
    # input_shapes=[(None, 3, None, None)],

    graph_author="CLIKA",
    graph_description="",
)
```

## References

<https://github.com/pytorch/examples/tree/main/mnist>
