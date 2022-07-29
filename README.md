# BuxLab

This is a [Flax](https://github.com/google/flax) implementation of parts of [BugLab](https://github.com/microsoft/neurips21-self-supervised-bug-detection-and-repair). Concretely, it implements a GNN-based model for detecting bugs in python code.

Limitations:
* The bug generation part of BugLab is not implemented. As the core rewrite selection model is the same, this would only require adding an appropriate objective and ensuring that batching works.
* The dynamic data generation of BugLab is not ported.
* The GREAT model for obtaining code representations as alternative to a GNN is not implemented.

## Running
Data needs to be obtained using the original BugLab data generation pipeline.

The model is configured via gin, using a configuration file for which a sample is included in `buxlab/buxlab.mini.config`.

To run training, call
```python
python -m buxlab.cli.train --data_dir PATH_TO_BUGLAB_DATA --gin_file buxlab/buxlab.config --result_dir RESULT_DIR
```