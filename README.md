# multimodal-coding
A project for data compression in multimodal models

# Using This Project

## Set Up Local Environment
First, you also need to allow `direnv` by using:

```bash
direnv allow
```

Then, to get started, run the following command to build set up your project.
```bash
make init
```

## Managing your Virtual Environments
We use `poetry` to manage virtual environments. For details read the [documentation here](https://python-poetry.org/docs/basic-usage). Some of the most common commands are listed below:

```bash
# Add dependencies
poetry add scikit-learn==2.1.2

# Install dependencies
make install

# Update dependencies
make update

# Activate local virtual environment
poetry shell

# Package code
make build 
```

## Run Experiments

#### 1. Build and deploy your container

Make sure your dependencies are properly defined in your `poetry.lock` configuration file. If you have made changes to it, do not forget to update your lock file by running:
```
make update
```
 To build your container image run:
```bash
make training-image
```

#### 2. Define entry point to launch experiments
Define entry points to your tasks in the `MLProject` file.

#### 3. Launch and Manage jobs
To launch and manage jobs using Slurm we created the following scripts: `slurm-launch`, `slurm-log`, `slurm-status`, and `slurm-cancel`. The documentation on how `slurm-launch` works is available by running:
```bash
slurm-launch --help
```

To launch jobs using the Kubernetes please use `k8s-launch`. Instructions for this script is available by running:
```bash
k8s-launch --help
```
To manage Kubernetes jobs we recommend to use [k9s](https://github.com/derailed/k9s).

