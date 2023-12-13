# Step Vulnerability Guided Mean Fluctuation Adversarial Attack  against Conditional Diffusion Models

<!-- <p align="center">
<img src=assets/results.gif />
</p> -->



**Step Vulnerability Guided Mean Fluctuation Adversarial Attack  against Conditional Diffusion Models
[Hongwei Yu](https://scholar.google.com.hk/citations?user=cDidt64AAAAJ&hl=zh-CN) Xinglong Ding**
<br/>


<p align="center">
<img src=assets/method.png />
</p>

## News

### November 2023
- We upload the demo of our work, you can using the upload adversarial examples to attack [LDM](https://github.com/CompVis/latent-diffusion).


  
## Requirements
A suitable [conda](https://conda.io/) environment named `MFA` can be created
and activated with:
We use the same environment with LDM.
```
conda env create -f environment.yaml
conda activate MFA
```

# Diffusion Models are sensitive to the mean value
<p align="center">
<img src=assets/meanshift.png />
</p>

We present the results of the comparison between MFA-MVS and the direct modification of input noise $x_T$

The figure shows that MFA-MVS is very similar to $X_T-0.2$, and -MFA-MVS(negating the loss) is very similar to $X_T+0.2$, which indicates that our method can effectively generate mean fluctuations and control the mean value.


# Test MFA(Mean Fluctuation Adversarial Attack)
## Inpainting Task
<p align="center">
<img src=assets/Inpaintgit.png />
</p>

### Download the pre-trained Inpainting Model
```
wget -O models/ldm/inpainting_big/last.ckpt https://heibox.uni-heidelberg.de/f/4d9ac7ea40c64582b7c9/?dl=1
```
### Using the provided adversarial examples to attack Inpainting task
```
python scripts/inpaint.py --indir adversarial_demo/ --outdir outputs/inpainting_results
```

## Super Resolution Task

<p align="center">
<img src=assets/SRgit2.png />
</p>

### Using the provided adversarial examples to attack Super Resolution task
```
python scripts/SuperResolution.py --indir adversarial_demo/superresolution 
```


## Coming Soon...

* The code of generating adversarial examples.

## Comments 

- Our codebase for the diffusion models builds heavily on [Latent Diffusion Models](https://github.com/CompVis/latent-diffusion).
Thanks for open-sourcing!




