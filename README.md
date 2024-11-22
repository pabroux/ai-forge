# <img width="20" height="20" src="https://gitlab.com/uploads/-/system/project/avatar/57187700/brain.png?width=96" alt="Banner"> AIBase

## About
AIBase gives you a codebase for any deep learning project. Don't reinvent the wheel!

## Table of Contents

- [Supported frameworks](#supported-frameworks)
- [Useful Commands](#useful-commands)

## Supported Frameworks
Here are the supported frameworks:
- Hugging Face
- Lightning
- Pytorch (Vanilla)
> ℹ️
> Each framework follows the pattern in which you pass to a `train.py` file a config file (e.g. `config_test.py`) containing your model, your dataset and everything required to train a model.


## Useful Commands
To launch a _MLflow_ server, in the `service/experimentManager/MLflow` folder, execute the following script:
```
❯ ./launch_server.sh
```