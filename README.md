<p align="center">
  <img width=375 src="https://github.com/user-attachments/assets/9e48f24f-bb22-4972-96e1-7b33c81430f6">
</p>
<p align="center">
  <b>A simple AI codebase</b>
</p>
<p align="center">
  <a href="https://github.com//Sulfyderz/AI-Forge/blob/master/LICENSE">
    <img src="https://img.shields.io/github/license/Sulfyderz/AI-Forge.svg" alt="License Badge">
  </a>
</p>


## About
AI-Forge gives you a codebase for any deep learning project.

## Table of Contents

- [Supported frameworks](#supported-frameworks)
- [Useful Commands](#useful-commands)

## Supported Frameworks
Here are the supported frameworks:
- Hugging Face
- Lightning
- Pytorch

> [!NOTE]
> Each framework follows the pattern in which you pass to a `train.py` file a config file (e.g. `config_test.py`) containing your model, your dataset and everything required to train a model. 


## Useful Commands
To launch a _MLflow_ server, in the `service/experimentManager/MLflow` folder, execute the following script:
```
./launch_server.sh
```
