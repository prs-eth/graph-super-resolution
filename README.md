# Learning Graph Regularisation for Guided Super-Resolution

This is the official implementation of the paper: 

### "Learning Graph Regularisation for Guided Super-Resolution" (CVPR 2022)

Riccardo de Lutio*, Alexander Becker*, Stefano D'Aronco, Stefania Russo, Jan Dirk Wegner and Konrad Schindler (&ast;equal contribution)

**[[Paper]](https://arxiv.org/abs/2203.14297) &nbsp;|&nbsp; [[5 minute explainer video]](https://www.youtube.com/watch?v=ZFDErHUlCBE&ab_channel=RiccardodeLutio)**

![Teaser](images/teaser_dark.png#gh-dark-mode-only)
![Teaser](images/teaser_light.png#gh-light-mode-only)

## Abstract

We introduce a novel formulation for guided super-resolution. Its core is a differentiable optimisation layer that operates on a learned affinity graph. The learned graph potentials make it possible to leverage rich contextual information from the guide image, while the explicit graph optimisation within the architecture guarantees rigorous fidelity of the high-resolution target to the low-resolution source. 
With the decision to employ the source as a constraint rather than only as an input to the prediction, our method differs from state-of-the-art deep architectures for guided super-resolution, which produce targets that, when downsampled, will only approximately reproduce the source. This is not only theoretically appealing, but also produces crisper, more natural-looking images.
A key property of our method is that, although the graph connectivity is restricted to the pixel lattice, the associated edge potentials are learned with a deep feature extractor and can encode rich context information over large receptive fields. By taking advantage of the sparse graph connectivity, it becomes possible to propagate gradients through the optimisation layer and learn the edge potentials from data.
We extensively evaluate our method on several datasets, and consistently outperform recent baselines in terms of quantitative reconstruction errors, while also delivering visually sharper outputs. Moreover, we demonstrate that our method generalises particularly well to new datasets not seen during training.

## Setup

### Dependencies
We recommend creating a new conda environment with all required dependencies by running
```bash
conda env create -f environment.yml
conda activate graph-sr
```

### Data
To reproduce our results, create a data directory (e.g. `./data`) with the three datasets:
* **NYUv2**: Download the labeled dataset from [[here]](https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html) and place the `nyu_depth_v2_labeled.mat` in `./data/NYU Depth v2`, along with our split indices file from [[here]](https://drive.google.com/file/d/1MclM7cejBAFBilZUJ4xCRBhzv6SJvJ7v/view?usp=sharing).
* **Middlebury**: Download the 2005-2014 scenes (full size, two-view) from [[here]](https://vision.middlebury.edu/stereo/data/) and place the extracted scenes in `./data/Middlebury/<year>/<scene>`. For the 2005 dataset, make sure to only put the scenes for which ground truth is available. The data splits are defined in code.
* **DIML**: Download the indoor data sample from [[here]](https://dimlrgbd.github.io) and extract it into `./data/DIML/{train,test}` respectively. Then run `python scripts/create_diml_npy.py ./data/DIML` to create numpy binary files for faster data loading.

### Checkpoints
Our pretrained model checkpoints which were used for the numbers in our paper, for all three datasets and upsampling factors, can be downloaded from [[here]](https://drive.google.com/drive/folders/17WgvuyoPnQPpOwIlzQSn20I8PB36bCzO?usp=sharing).

## Training

Run the training script via
```bash
python run_train.py --dataset <...> --data-dir <...> --save-dir <...>
```
Hyperparameter defaults are set to the values from the paper. Depending on the dataset, you have to adjust the number of epochs (`--num-epochs`) and the scheduler step size (`--lr-step`), see appendix A of the paper. You can see all available training options by running 
```bash
python run_train.py -h
```

## Evaluation

For test set evaluation, run

```bash
python run_eval.py --checkpoint <...> --dataset <...> --data-dir <...>
```
Again, you can query all available options by running 
```bash
python run_train.py -h
```

## Citation
```
@inproceedings{deLutio2022,
 author = {de Lutio, Riccardo and Becker, Alexander and D'Aronco, Stefano and Russo, Stefania and Wegner, Jan D. and Schindler, Konrad},
 title = {Learning Graph Regularisation for Guided Super-Resolution},
 booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
 year = {2022},
} 
```
