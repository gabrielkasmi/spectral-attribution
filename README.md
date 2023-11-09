# Assessment of the Reliablity of a Model's Decision by Generalizing Attribution to the Wavelet Domain

Gabriel Kasmi, Laurent Dubus, Yves-Marie Saint-Drenan, Philippe Blanc

Accepted for the [XAI in action workshop at NeurIPS 2023](https://xai-in-action.github.io/), Dec 16th 2023 in New-Orleans. Access the paper [here](https://arxiv.org/abs/2305.14979).

## Overview 

Deep neural networks are widely used in computer vision, but there is a growing consensus that their deployment in real-world applications can be problematic due to reliability issues. This is primarily attributed to their black-box nature, leading to the need for explainable AI (XAI) techniques for better understanding model decisions, and distribution shifts that can cause unpredictable failures. To safely deploy deep learning models, there is a requirement for tools that can audit both the relevance and robustness of a model's decisions in the face of distribution shifts. While attribution methods have helped understand the decision process, and Fourier analysis has been used to assess robustness, there is currently a research gap in combining these approaches to audit decisions in both the pixel (space) and frequency (scale) domains, which would provide a more comprehensive assessment

We introduce **Wavelet Scale Attribution Method (WCAM)**, a novel attribution method that represents a model decision in the space-scale (or wavelet) domain. The decomposition in the space-scale domain highlights which structural components (textures, edges, shapes) are important for the prediction, allowing us to assess the *relevance* of a decision process. Moreover, as scales correspond to frequencies, we simultaneously evaluate whether this decision process is robust. We discuss the potential of WCAM for application in expert settings (e.g., medical imaging or remote sensing), show that we can quantify the robustness of a prediction with WCAM, and highlight concerning phenomena regarding the consistency of the decision process of deep learning models.


<p align="center">
<img src="https://github.com/gabrielkasmi/spectral-attribution/blob/main/assets/flowchart-wcam.png" width=500px>
</p>

## Usage

If you want to use the source code of the WCAM, we recommend you create a virtual environment. This can be done as follows:

```python
conda create --name spectral_attribution
conda activate spectral_attribution
pip install -r requirements.txt
```


### An illustrative example

If you want to see an example of how the WCAM works, you can run the notebook `example.ipynb`. This notebooks guides you through the main steps to load an image, compte its WCAM and visualize the results. 

### Source code

The source code of the spectral attribution method is located in the folder `spectral_sobol`. This source code is based on Fel et. al.'s Sobol attribution method [1]. Do not hesitate to use it in your projects !

## Main results

### Benchmarks compared to other methods

We evaluate our method against a range of popular methods and across various model architectures. On the $\mu$-Fidelity of Bhatt *et al.* [2], we outperform existing black-box methods and are competitive with white-box attribution methods. The projection in the space-scale domain is the cause for the superiority of our method: we can see that the WCAM shows that the coarser scales are essential for a prediction. When flattening the WCAM accross scales to derive the Spatial WCAM, our method performs similarly to other attribution methods.

<p align="center">
<img src="https://github.com/gabrielkasmi/spectral-attribution/blob/main/assets/figures/mu_fidelity.png" width=500px>
</p>


### Assessing the reliability of a prediction

Scales in the wavelet domain correspond to dyadic frequency ranges in the Fourier domain. The smallest scales correspond to the highest frequencies. Therefore, the WCAM connects attribution with frequency-centric approaches to model robustness. We can use the WCAM to quantify the reliance on various frequency ranges. We dinstinguish three types of models, the "standard" (ST), corresponding to the vanilla ERM, the "robust" (RT), corresponding to models trained with techniques such as AugMix or PixMix and the "adversarial" (AT) corresponding to models that underwent adversarial training (e.g., Fast-ADV). Earlier works (e.g., [3]) showed that robust and adversarial models rely on lower frequency components than the vanilla models. The WCAM enables us to recover this result, and thus to quantify the robustness of a model and its individual predictions.

<p align="center">
<img src="https://github.com/gabrielkasmi/spectral-attribution/blob/main/assets/figures/robustness.png" width=500px>
</p>

## License and citation 

### License

This work is provided under GNU License.

### Citation

```
@inproceedings{kasmi2023assessment,
  title={Assessment of the Reliablity of a Model's Decision by Generalizing Attribution to the Wavelet Domain},
  author={Kasmi, Gabriel and Dubus, Laurent and Saint-Drenan, Yves-Marie and BLANC, Philippe},
  booktitle={XAI in Action: Past, Present, and Future Applications},
  year={2023}
}
```

Like this work ? Do not hesitate to star us !

## References

[1] Fel, T., Cadène, R., Chalvidal, M., Cord, M., Vigouroux, D., & Serre, T. (2021). Look at the variance! efficient black-box explanations with sobol-based sensitivity analysis. *Advances in Neural Information Processing Systems*, 34, 26005-26014.

[2] Bhatt, U., Weller, A., & Moura, J. M. (2020). Evaluating and aggregating feature-based model explanations. *International Joint Conference on Artificial Intelligence*. 

[3] Chen, Y., Ren, Q., & Yan, J. (2022). Rethinking and Improving Robustness of Convolutional Neural Networks: a Shapley Value-based Approach in Frequency Domain. *Advances in Neural Information Processing Systems*, 35, 324-337.
