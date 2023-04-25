# Scale matters: attribution meets the spectral domain to explain model brittleness to image corruptions

Gabriel Kasmi, Philippe Blanc, Yves-Marie Saint-Drenan, Laurent Dubus

## Overview 

Spectral attribution methods consist in perturbing images in the space-scale domain or frequency domain to recover the regions in these domains that contribute to a model's prediction. It thus brigde the gap between existing attribution methods that only identify the important features in the spatial domain and robustness studies that have quantitatively asserted that robust models tend to rely more on low frequency components than standard models. 

Our first main contribution is the <b> wavelet class activation map </b> (WCAM), which highlights the important regions for the prediction of a label in the space-scale domain. WCAMs are obtained by computing the Sobol indices coming from a Sobol sequence. These sequence is converted as a mask, which is applied to the wavelet transform of the input image. Perturbed images are obtained by inverting the perturbed wavelet transform of the image. As an output, WCAM highlight the important regions (according to their Sobol indices) of regions in the space-scale domain. Besides, WCAMs can be wrapped to obtain the importance of the various image regions. WCAMs are based on the Sobol attribution method introcued by [Fel et al. (2021)](https://openreview.net/forum?id=hA-PHQGOjqQ).

<p align="center">
<img src="https://github.com/gabrielkasmi/spectral-attribution/blob/main/assets/flowchart-wcam.png" width=500px>
</p>

## Main results

### Models rely on a limited set of wavelet coefficients to make predictions

WCAM addresses two questions: <i> how perturbations affect a model </i> and <i> how robustness mitigation strategies improve the robustness to distribution shifts </i>. Through experiments on ImageNet-C and ImageNet-E we show that models rely on a limited set of coefficients to make their prediction. This behavior is consistant across model backbones (convolutional, transformers and self-supervised) and training approaches (standard, robust and adversarial). 

### The reconstruction depth is predictive of the robustness of a prediction

It is of great interest for practitioners to target predictions that are likely to be erroneous. To this end and based on our findings, we introduce the <i> reconstruction depth </i> a measure based on the number of wavelet coefficients necessary to reconstruct the sufficient image (i.e., an image that is correctly classified by the model). Experiments show a correlation between the robustness of a prediction and the reconstruction depth. This pattern is consistent across backbones and training strategies. Besides, it also holds if the reference image is a perturbed image. This paves the way for leveraging the reconstruction depth as a predictive measure of the robustness of a prediction, which can be very useful in practical settings for flagging unreliable model predictions. Moreover, this shows that WCAMs provide a way of asserting instance-based robustness whereas existing studies are limited to model-based assessments. 

## Usage

If you want to use the source code of the Spectral attribution methods or replicate our results, we recommend you create a virtual environment. This can be done as follows:

```python
conda env create -file spectral_attribution.yml
conda activate spectral_attribution
```

### Using the spectral attribution methods (WCAM and Fourier-CAM)

The source code of the spectral attribution method is located in the folder `spectral_sobol`. This source code is based on Fel et. al.'s [Sobol attribution method](https://proceedings.neurips.cc/paper/2021/hash/da94cbeff56cfda50785df477941308b-Abstract.html) (NeurIPS 2021). You can access a demo of this attribution method in the notebook `example.ipynb`.

### Replication of our results

The folders `scripts` and `notebooks` contain the scripts and notebook used to generate the results presented in the paper. In addition to these scripts and notebooks, you will need to download the ImageNet validation set (accessible [here](https://www.image-net.org/download.php)) and the supplementary material provided in our [Zenodo repository](). 

## License and citation 

### License

### Citation

```
@article{

}
```
