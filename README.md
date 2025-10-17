# <img width="35" height="35" alt="image" src="https://github.com/user-attachments/assets/2700a971-e5d6-4036-b03f-2f89c9791609" /> TorchForge

#### A PyTorch-native agentic RL library that lets you focus on algorithms—not infra.
[![Unit Tests](https://github.com/meta-pytorch/forge/actions/workflows/unit_test.yaml/badge.svg?branch=main)](https://github.com/meta-pytorch/forge/actions/workflows/unit_test.yaml?query=branch%3Amain)
[![GPU Tests](https://github.com/meta-pytorch/forge/actions/workflows/gpu_test.yaml/badge.svg?branch=main)](https://github.com/meta-pytorch/forge/actions/workflows/gpu_test.yaml?query=branch%3Amain)

## Overview
The primary purpose of the TorchForge ecosystem is to delineate infra concerns from model concerns thereby making RL experimentation easier. TorchForge delivers this by providing clear RL abstractions and one scalable implementation of these abstractions. When you need fine-grained control over placement, fault handling/redirecting training loads during a run, or communication patterns, the primitives are there. When you don’t, you can focus purely on your RL algorithm.

Key features:
- Usability for rapid research (isolating the RL loop from infrastructure)
- Hackability for power users (all parts of the RL loop can be easily modified without interacting with infrastructure)
- Scalability (ability to shift between async and synchronous training and across thousands of GPUs)

> ⚠️ **Early Development Warning** TorchForge is currently in an experimental
> stage. You should expect bugs, incomplete features, and APIs that may change
> in future versions. The project welcomes bugfixes, but to make sure things are
> well coordinated you should discuss any significant change before starting the
> work. It's recommended that you signal your intention to contribute in the
> issue tracker, either by filing a new issue or by claiming an existing one.

## 📖 Documentation (Coming Soon)

View TorchForge's hosted documentation (coming soon)

## Tutorials

You can also find our notebook tutorials (coming soon)

## Installation

### Basic

TorchForge requires the latest PyTorch nightly with [Monarch](https://github.com/meta-pytorch/monarch), [vLLM](https://github.com/vllm-project/vllm), and [torchtitan](https://github.com/pytorch/torchtitan). For convenience,
we have pre-packaged these dependencies as wheels in assets/wheels. (Note that the basic install script
uses [DNF](https://docs.fedoraproject.org/en-US/quick-docs/dnf/), but could be easily extended to other Linux OS.)

TorchForge requires the Github CLI (gh) to download a compatible vLLM package. See [here](https://github.com/cli/cli#installation) for gh install instructions before continuting. Please login to gh with your Github account before continuing with `gh auth login`. You may use either https or ssh as the protocol for authentication.

```bash
conda create -n forge python=3.10
conda activate forge
./scripts/install.sh
```

Optional: By default, the packages installation uses conda. If user wants to install system packages on the target machine instead of conda, they can pass the `--use-sudo` to the installation script: `./script/install.sh --use-sudo`.

After install, you can run the following command and should see output confirming GRPO training is running (you need a minimum 3 GPU devices):

```
python -m apps.grpo.main --config apps/grpo/qwen3_1_7b.yaml
```

If you need to re-build the wheels for whatever reason, you can do so with:
```bash
./scripts/build_wheels.sh
```

## Quick Start

To run SFT on a Llama3 8B model, run

```bash
python -m apps.sft.main --config apps/sft/llama3_8b.yaml
```

### Citation

## License

Source code is made available under a [BSD 3 license](./LICENSE), however you may have other legal obligations that govern your use of other content linked in this repository, such as the license or terms of service for third-party data and models.
