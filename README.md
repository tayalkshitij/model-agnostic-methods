# Model-agnostic-methods for Text Classification

This repository contains the code to reproduce the results of the paper (**Applying Model-agnostic Methods to Handle Inherent Noise in Large Scale Text Classification**) currently in review at KDD 2020. 


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
git clone git@github.com:facebookresearch/access.git
cd access
pip install -e .
```

### How to use

Evaluate the pretrained model on WikiLarge:
```
python scripts/evaluate.py
```

Simplify text with the pretrained model
```
python scripts/generate.py < my_file.complex
```

Train a model
```
python scripts/train.py
```

## Dataset

Automotive Dataset [here](https://drive.google.com/open?id=1w2YuR1knf3LJPfq6wVxo3frHlbqP6BqA).
Beauty Dataset [here](https://drive.google.com/open?id=1rNNSdGvMcivdXI72SGvrd_Td1Ah2_Vo7).
Electronics Dataset [here](https://drive.google.com/open?id=1qDX8gBWrG0pdqqkJS8HUtI6K3kKtKfuR).



The model's output simplifications can be viewed on the [EASSE HTML report](http://htmlpreview.github.io/?https://github.com/facebookresearch/access/blob/master/system_output/easse_report.html).


## Author

If you have any question, please contact the author:
**Kshitij Tayal** ([tayal007@umn.edu](mailto:tayal007@umn.edu))

## License

See the [LICENSE](LICENSE) file for more details.
