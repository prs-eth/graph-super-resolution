# Learning Graph Regularisation for Guided Super-Resolution

This is the official implementation of our CVPR 2022 paper by Riccardo de Lutio*, Alexander Becker*, Stefano D'Aronco, Stefania Russo, Jan Dirk Wegner and Konrad Schindler (&ast;equal contribution). [[Project Page]](https://example.com/) | [[Paper]](https://arxiv.org/)

![Teaser](images/teaser_dark.png#gh-dark-mode-only)
![Teaser](images/teaser_light.png#gh-light-mode-only)

## Abstract

We introduce a novel formulation for guided super-resolution. Its core is a differentiable optimisation layer that operates on a learned affinity graph. The learned graph potentials make it possible to leverage rich contextual information from the guide image, while the explicit graph optimisation within the architecture guarantees exact fidelity of the high-resolution target to the low-resolution source. 
With the decision to employ the source as a constraint rather than only as an input to the prediction, our method differs from state-of-the-art deep architectures for guided super-resolution, which produce targets that, when downsampled, will only approximately reproduce the source. This is not only theoretically appealing, but also produces crisper, more natural-looking images.
A key property of our method is that, although the graph connectivity is restricted to the pixel lattice, the associated edge potentials are learned with a deep feature extractor and can encode rich context information over large receptive fields. By taking advantage of the sparse graph connectivity, it becomes possible to propagate gradients through the optimisation layer and learn the edge potentials from data.
We extensively evaluate our method on several datasets, and consistently outperform recent baselines in terms of quantitative reconstruction errors, while also delivering visually sharper outputs. Moreover, we demonstrate that our method generalises particularly well to new datasets not seen during training.

## Setup

We recommend creating a new conda environment with all required dependencies by running
```bash
conda env create -f environment.yml
conda activate graph-sr
```

## Training

Run the training script via
```bash
python run_train.py --dataset <...> --data-dir <...> --save-dir <...>
```
You can see all available training options by running 
```bash
python run_train.py -h
```

## Reproducing our results
Model checkpoints can be downloaded from [here]. 

## Citation
```
@inproceedings{deLutio2022,
 author = {de Lutio, Riccardo and Becker, Alexander and D'Aronco, Stefano and Russo, Stefania and Wegner, Jan D. and Schindler, Konrad},
 title = {Learning Graph Regularisation for Guided Super-Resolution},
 booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
 year = {2022},
} 
```
