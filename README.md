# Prompt-Guided Prefiltering for VLM Image Compression

**Abstract:**
The rapid progress of large Vision Language Models (VLM) has enabled a wide range of applications, such as image understanding and Visual Question Answering (VQA). Query images are often uploaded to the cloud, where VLMs are typically hosted, hence efficient image compression becomes crucial. However, traditional human-centric codecs are suboptimal in this setting because they preserve many task-irrelevant details. Existing Image Coding for Machines (ICM) methods also fall short, as they assume a fixed set of downstream tasks and cannot adapt to prompt-driven VLMs with an open-ended variety of objectives. We propose a lightweight, plug-and-play, prompt-guided prefiltering module to identify image regions most relevant to the text prompt, and consequently to the downstream task. The module preserves important details while smoothing out less relevant areas to improve compression efficiency. It is codec-agnostic and can be applied before conventional and learned encoders.
Experiments on several VQA benchmarks show that our approach achieves a 25–50\% average bitrate reduction while maintaining the same task accuracy.

![Visual Results](figures/vis_results.png)

## Citation ##
This is the official code release of the ["Prompt-Guided Prefiltering for VLM Image Compression"](https://arxiv.org/abs/2604.00314) paper. It is under the MIT license. Please cite it as follows:

```
@article{azizian2026_pgp_vlm_compression,
  title     = {Prompt-Guided Prefiltering for VLM Image Compression},
  author    = {Bardia Azizian and Ivan V. Baji{\'c}},
  journal   = {arXiv preprint arXiv:2604.00314},
  year      = {2026},
  eprint    = {2604.00314},
  archivePrefix = {arXiv},
  primaryClass = {eess.IV}
}
```


## Dependencies and Third-Party Code

This project leverages several third-party tools and repositories:

- **TinyCLIP**  
  Portions of our codebase utilize components from the [TinyCLIP model](https://github.com/wkcn/TinyCLIP). For more information, please visit the official TinyCLIP repository.

- **Cheng 2020 Model Compression**  
  For implementing [Cheng et al., 2020](https://openaccess.thecvf.com/content_CVPR_2020/html/Cheng_Learned_Image_Compression_With_Discretized_Gaussian_Mixture_Likelihoods_and_Attention_CVPR_2020_paper.html) model-based compression, we use the [CompressAI](https://github.com/interdigitalinc/compressai) library.

- **VVC (H.266) Compression**  
  To support image compression using the H.266/VVC standard, this project relies on the [VVenC](https://github.com/fraunhoferhhi/vvenc) encoder. Please make sure both `vvencapp` and `vvdecapp` executables are installed and available in your `PATH`, as described in the VVenC installation documentation.



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

