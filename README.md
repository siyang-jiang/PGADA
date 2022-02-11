# FewShiftBed

## Enviroment
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

Configure your experiment by changing the values in `configs/*.py`, then launch your experiment.
```python -m scripts.run_experiment```

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

Image perturbations are modified from https://github.com/hendrycks/robustness

