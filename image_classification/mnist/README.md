# MNIST Compressing Example

<ins>CLIKA SDK</ins> example to compress CNN model on `MNIST` dataset

# Requirements

- Python >= 3.8
- Install PyTorch [pytorch.org](https://pytorch.org/)
- `pip install -r image_classification/mnist/requirements.txt`

# Prepare Dataset

dataset will be automatically downloaded using PyTorch into `image_classification/mnist/MNIST` when running `mnist_main.py` the first
time

after running the `mnist_main.py` script the first time, the dataset directory tree should look like the following:

```
image_classification/mnist/MNIST/
├── raw/
│   ├── t10k-images-idx3-ubyte
│   ├── t10k-images-idx3-ubyte.gz
│   ├── t10k-images-idx1-ubyte
│   ├── t10k-images-idx1-ubyte.gz
...

```

# Run Examples

```
# pwd: image_classification/mnist
python3 mnist_main.py
```

# References

https://github.com/pytorch/examples/tree/main/mnist
