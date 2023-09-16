# Scale Matters: Attribution Meets the Wavelet Domain to Explain Model Sensitivity to Image Corruptions

Anonymous submission 

## Overview 

Spectral attribution methods consist in perturbing images in the space-scale domain or frequency domain to recover the regions in these domains that contribute to a model's prediction. It thus brigde the gap between existing attribution methods that only identify the important features in the spatial domain and robustness studies that have quantitatively asserted that robust models tend to rely more on low frequency components than standard models. 

Our main contribution is the <b> wavelet scale attribution method </b> (WCAM), which highlights the important regions for the prediction of a label in the space-scale domain. WCAMs are obtained by computing the Sobol indices coming from a Sobol sequence. These sequence is converted as a mask, which is applied to the wavelet transform of the input image. Perturbed images are obtained by inverting the perturbed wavelet transform of the image. As an output, WCAM highlight the important regions (according to their Sobol indices) of regions in the space-scale domain. Besides, WCAMs can be wrapped to obtain the importance of the various image regions. WCAMs are based on the Sobol attribution method introcued by [Fel et al. (2021)](https://openreview.net/forum?id=hA-PHQGOjqQ).

<p align="center">
<img src="https://github.com/gabrielkasmi/spectral-attribution/blob/main/assets/flowchart-wcam.png" width=500px>
</p>

## Usage

If you want to use the source code of the Spectral attribution methods or replicate our results, we recommend you create a virtual environment. This can be done as follows:

```python
conda create --name spectral_attribution
conda activate spectral_attribution
pip install -r requirements.txt
```

### Using the spectral attribution methods (WCAM and Fourier-CAM)

The source code of the spectral attribution method is located in the folder `spectral_sobol`. This source code is based on Fel et. al.'s [Sobol attribution method](https://proceedings.neurips.cc/paper/2021/hash/da94cbeff56cfda50785df477941308b-Abstract.html) (NeurIPS 2021). You can access a demo of this attribution method in the notebook `example.ipynb`.


## Main results

### Models rely on a limited set of wavelet coefficients to make predictions

We consider two cases of corruption: when corruption does not change the prediction and when corruption changes the prediction. The first element to notice is that the model focuses on a limited number of regions in the space-scale domain. Higher scale coefficients (i.e., high frequencies) hardly contribute to the prediction. Even at larger scales, the model focuses on a few hot spots. This tendency is common through models (vanilla ResNet, robust and adversarial models as well as vision transformers). The consequence of a perturbation is to alter the importance of the coefficients, until the model ultimately makes an incorrect prediction.

<p align="center">
<img src="https://github.com/gabrielkasmi/spectral-attribution/blob/main/assets/corruptions_baseline_2-1.png" width=500px>
</p>


### WCAMs highlight the sufficient information for image classification 

We show that we can extract a <i> sufficient image </i> using the importance coefficients. The sufficient image is reconstructed using only a subset of the most important wavelet coefficients. In our examples, we can see that for the image of a cat, the model needs detailed information around the eyes, which is not the case for the fox. In both cases, we also see that the models do not need information in the background, as we can completely hide it without changing the prediction. Identifying a sufficient image could have numerous applications, for instance, in transfer compressed sensing. In some applications (e.g., medical imaging), data acquisition is expensive. 

<p align="center">
<img src="https://github.com/gabrielkasmi/spectral-attribution/blob/main/assets/reconstruction-sufficient-2-1.png" width=600px>
</p>

### WCAMs show that zoom improves performance mainly because it discards background information, not because it brings new information

As seen from the table below, the importance of the level introduced by the zoom is negligible. Herefore, the higher accuracy comes from the fact that the object is more prominent than before and there is less background on the image, not because the model finds new features on the object. We can also see that adversarial models rely less on high frequencies (or higher levels) than robust models. 

<p align="center">
<img src="https://github.com/gabrielkasmi/spectral-attribution/blob/main/assets/table.png" width=800px>
</p>

### Replication of our results

The folders `scripts` and `notebooks` contain the scripts and notebook used to generate the results presented in the paper. In addition to these scripts and notebooks, you will need to download:
* The ImageNet validation set (accessible [here](https://www.image-net.org/download.php))
* ImageNet-A (accessible [here](https://github.com/hendrycks/natural-adv-examples)) and ImageNet-R (accessible [here](https://github.com/hendrycks/imagenet-r))
* The supplementary data and models accessible on our Zenodo repository, accessible [here](https://doi.org/10.5281/zenodo.7924905)

## License and citation 

### License

This work is provided under GNU License.

### Citation

```
@article{kasmi2023scale,
  title={Scale Matters: Attribution Meets the Wavelet Domain to Explain Model Sensitivity to Image Corruptions},
  author={Kasmi, Gabriel and Dubus, Laurent and Drenan, Yves-Marie Saint and Blanc, Philippe},
  journal={arXiv preprint arXiv:2305.14979},
  year={2023}
}
```
