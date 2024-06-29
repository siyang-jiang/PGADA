# PGADA Offical Source Code
#### PGADA: Perturbation-Guided Adversarial Alignmentfor Few-shot Learning Under the Support-Query Shift (PAKDD 22)

- [[Paper 🤗](https://arxiv.org/abs/2205.03817)]  
## News

- **2024-06-30** Release Sample Code🔥
- **2022-05-19** **Best Student Paper Award** in PAKDD 2022🔥🔥🔥
- **2022-01-15** PGADA is accepted by **PAKDD 2022**🔥


## Enviroment
 - GPU > 8G (>=24G for mini-Imagenet)
 - Python 3.7
 - [Pytorch](http://pytorch.org/) 1.7
 - CUDA 10

## Getting started

```
virtualenv venv --python=python3.7
source venv/bin/activate
```

Then install dependencies: `pip install -r requirements.txt`
Some perturbations used in CIFAR-100-C-FewShot and *mini*ImageNet-C use Wand: `sudo apt-get install libmagickwand-dev`

## Data
To install the datasets to your machine, please follow [this walkthrough](DATASETS.md).

## Run an experiment

Configure your experiment by changing the values in `configs/*.py`, then launch your experiment. (Make Sure all config are right)
```python -m scripts.erm_training```

Testing
```python -m scripts.eval_model```


On some machines, the `src` module will not be found by Python. If this happens to you, run
`export PYTHONPATH=$PYTHONPATH:path/to/FewShiftBed` to tell Python where you're at.

All outputs of the experiment (explicit configuration, logs, trained model state and TensorBoard logs) 
can then be found in the directory specified in `configs/experiment_config.py`. By default, an error will be risen if 
the specified directory already exists (in order to not harm the results of previous experiments). You may
change this behaviour in `configs/experiment_config.py` by setting `OVERWRITE = True`.

### Reproducing results

See the detailed documentation [here](REPRODUCING.md).

### Track trainings with Tensorboard

We log the loss and validation accuracy during the training for visualization in Tensorboard. The logs of an
experiment can be found in the output directory (`events.out.tfevents.[...]`). To visualize them in Tensorboard, run:
```
tensorboard --logdir=output_dir
```


## References
PGADA code is modified from https://github.com/ebennequin/meta-domain-shift and image perturbations are modified from https://github.com/hendrycks/robustness

## Cite this Work:
```
@inproceedings{jiang2022pgada,
  title={PGADA: Perturbation-Guided Adversarial Alignment for Few-Shot Learning Under the Support-Query Shift},
  author={Jiang, Siyang and Ding, Wei and Chen, Hsi-Wen and Chen, Ming-Syan},
  booktitle={Advances in Knowledge Discovery and Data Mining: 26th Pacific-Asia Conference, PAKDD 2022, Chengdu, China, May 16--19, 2022, Proceedings, Part I},
  pages={3--15},
  year={2022}
}
```

