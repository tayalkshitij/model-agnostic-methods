# Model-agnostic-methods for Text Classification

This repository contains the code to reproduce the results of the paper (**Applying Model-agnostic Methods to Handle Inherent Noise in Large Scale Text Classification**) accepted at COLING 2020. 


## Getting Started

### Dependencies

* Python 3.7
* Pandas
* Numpy
* Keras
* SkLearn
* Pickle

### Installing

```
git clone https://github.com/tayalkshitij/model-agnostic-methods.git
cd model-agnostic-methods/
```

### How to use

Train main model:
```
python code/main_experiment/main.py <path_to_dataset>
```

Train noise model:
```
python code/noise_experiment/noise_main.py <path_to_dataset>
```

## Dataset

Google drive link for the datasets are as follow:

Automotive Dataset [Link](https://drive.google.com/open?id=1w2YuR1knf3LJPfq6wVxo3frHlbqP6BqA).
Beauty Dataset [Link](https://drive.google.com/open?id=1rNNSdGvMcivdXI72SGvrd_Td1Ah2_Vo7).
Electronics Dataset [Link](https://drive.google.com/open?id=1qDX8gBWrG0pdqqkJS8HUtI6K3kKtKfuR).

## Pre-trained Embeddings

Glove [Link](https://nlp.stanford.edu/projects/glove/).

## Author

If you have any question, please contact the author:
**Kshitij Tayal** ([tayal007@umn.edu](mailto:tayal007@umn.edu))

## License

See the [LICENSE](LICENSE) file for more details.
