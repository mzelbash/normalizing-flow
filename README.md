# Normalizing Flow Lab

**Watch a neural network turn a picture into a Gaussian, then run it backward to create new samples.**

▶ **Live demo:** https://mzelbash.github.io/normalizing-flow/

Flow Lab is a one-page browser app for learning how **normalizing flows** work. It trains a real flow (RealNVP-style affine coupling layers) in your browser, in a few seconds, and lets you see every step: the data being bent into a bell curve, the exact density the model learns, and brand-new samples coming back out.

It was built as a teaching companion for **SEAS 8525: Computer Vision and Generative AI** at the George Washington University (Dr. Elbasheer).

---

## What you can do

1. **Pick a picture.** Choose a built-in shape (Smiley, Two moons, Spiral, Heart), type a word, or upload your own image. The dark pixels become a cloud of 2-D points: this is the data distribution p(x).
2. **Train.** Press **Train** and watch the negative log-likelihood fall. This is the exact loss of a normalizing flow, with no bound and no ELBO.
3. **Normalize (x → z).** Drag the slider from x to z, or press play, and watch the shape turn into a Gaussian one coupling layer at a time.
4. **Generate (z → x).** Switch to **Generate**. Fresh Gaussian noise runs backward through the same layers and becomes new samples shaped like your picture.
5. **Experiment.** Change the number of coupling layers (4, 8 or 12) and retrain. What can a smaller flow not do?

## What the panels show

| Panel | What it means |
|---|---|
| **Stage** (center) | The points after each layer. Normalize mode moves data x toward z; Generate mode moves noise z back toward x. |
| **Exact density p(x)** | The density the model has learned, computed for every pixel with the change-of-variables formula. |
| **Negative log-likelihood** | The training loss, in nats per point, with a small curve of its history. Lower is better. |
| **Lossless round trip** | The largest error of f⁻¹(f(x)) − x over the data. It is about 10⁻¹⁵: just computer rounding, so the flow really is invertible. |
| **Parameters** | The number of trainable weights in the flow. |

## How it works

Every flow layer is invertible and has a cheap Jacobian determinant, so the model gives an **exact** log-likelihood:

```
log p(x) = log p_z(f(x)) + log |det J_f(x)|
```

Each layer:

1. **Rotates** the plane by a fixed angle, so successive layers mix the two coordinates in different directions.
2. Applies an **affine coupling** step: one coordinate is copied, and the other is scaled and shifted by amounts computed from the first:

   ```
   y₁ = x₁
   y₂ = x₂ · exp(s(x₁)) + t(x₁)
   ```

   Here s and t come from a small neural network (two hidden layers of 32 units). The inverse is one line of algebra, and log |det J| is simply s(x₁).

Training details:

* Optimizer: Adam with a cosine learning-rate schedule
* Batch size 128, 3,500 steps
* The last layer of every s/t network starts at zero, so each coupling layer starts as the identity (it does nothing until it learns)
* Gradients are computed by hand (no ML library); training runs in a Web Worker so the page stays responsive

## Running it locally

There is nothing to install and no build step. The app is a single self-contained HTML file with no external dependencies.

```bash
git clone https://github.com/mzelbash/normalizing-flow.git
cd normalizing-flow
# open index.html in any modern browser, or serve the folder:
python3 -m http.server 8000
# then visit http://localhost:8000
```

## Keyboard shortcuts

| Key | Action |
|---|---|
| Space | Play through the layers |
| ← / → | Step one layer back or forward |

## Suggested classroom walk-through (about 10 minutes)

1. Start with **Two moons** and press **Train**.
2. Point out that the loss is the exact negative log-likelihood.
3. Drag the slider from x to z and stop at a few layers along the way.
4. Switch to **Generate** and compare the new samples with the original picture.
5. Retrain with **4 layers** and discuss what changes.
6. Point to the round-trip error: the flow is invertible to about 10⁻¹⁵.

## Learn more

* L. Dinh, D. Krueger, Y. Bengio. *NICE: Non-linear Independent Components Estimation.* 2014.
* D. Rezende, S. Mohamed. *Variational Inference with Normalizing Flows.* ICML 2015.
* L. Dinh, J. Sohl-Dickstein, S. Bengio. *Density Estimation using Real NVP.* ICLR 2017.
* D. P. Kingma, P. Dhariwal. *Glow: Generative Flow with Invertible 1×1 Convolutions.* NeurIPS 2018.
* G. Papamakarios et al. *Normalizing Flows for Probabilistic Modeling and Inference.* JMLR 2021 (a good review).

## License

Add your preferred license here (for example, MIT).
