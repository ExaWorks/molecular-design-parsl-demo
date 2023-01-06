# ML-in-the-loop molecular design with Parsl

[![Binder](http://mybinder.org/badge.svg)]([http://mybinder.org/v2/gh/binder-examples/r_with_python/master?urlpath=lab](https://mybinder.org/v2/gh/Parsl/parsl-tutorial/master))

This repository contains a tutorial showing how Parsl can be used to write a machine-learning-guided search for high-performing molecules.

The objective of this application is to identify which molecules have the largest ionization energies (IE, the amount of energy required to remove an electron). 

IE can be computed using various simulation packages (here we use [xTB](https://xtb-docs.readthedocs.io/en/latest/contents.html)); however, execution of these simulations is expensive, and thus, given a finite compute budget, we must carefully select which molecules to explore. 

In this example, we use machine learning to predict molecules with high IE based on previous computations (a process often called [active learning](https://pubs.acs.org/doi/abs/10.1021/acs.chemmater.0c00768)). We iteratively retrain the machine learning model to improve the accuracy of predictions. 

## Installation

The demo uses a few codes that are easiest to install with Anaconda. Our environment should work on both Linux and OS X can can be installed by:

```bash
conda env create --file environment.yml
```


## Tutorial

The notebook steps through the various phases of the workflow. 
1. Invoking the xTB simulation using Parsl such that it can run in parallel on configured resources (from several cores on a laptop through to thousands of nodes on a supercomputer). 
2. Developing a machine learning workflow can be created to train a model, use it to infer IE, and then combine inferences into a single data frame.
3. Combining the simulation and machine learning workflow such that the machine learning model is iteratively retrained and applied to the search space, with the highest value candidates then being prioritized for simulation, and the resulting simulations being used to retrain the model.
