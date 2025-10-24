# DCGAN Face Generator

This project trains a **Deep Convolutional Generative Adversarial Network (DCGAN)** to create new human faces based on the CelebA dataset. The goal of this README is to explain the big idea in approachable language, so you can understand what the model is doing even if you never read the code.

---

## 1. The Idea Behind GANs

A **Generative Adversarial Network (GAN)** is like a friendly art contest between two neural networks:

- The **Generator** is the artist. It starts with random noise and tries to draw a convincing face.
- The **Discriminator** is the judge. It looks at a face and decides if it is a real photo or a fake one from the generator.

Every training step is a round of this contest:

1. Show the discriminator a batch of real faces. It should answer “real”.
2. Ask the generator to invent a batch of fake faces from random noise.
3. Show those fakes to the discriminator. It should answer “fake”.
4. If the discriminator spots the fake, the generator studies the feedback to improve.
5. If the generator fools the discriminator, the discriminator studies that fake so it can do better next time.

Over many rounds, the generator becomes skilled at inventing faces that look realistic, and the discriminator becomes a sharp critic. Training ends when the two opponents reach a balance—fakes look real enough that the discriminator struggles to tell them apart.

---

## 2. A Gentle Peek At The Math

At the heart of this contest is a simple goal: real images should be scored near **1**, fake images near **0**. Both networks rely on the same score, which is computed using **binary cross-entropy loss** (a standard way to measure how close a predicted probability is to the desired answer).

The classic formulation is a min–max game:

```text
Discriminator: maximize log(D(real)) + log(1 - D(fake))
Generator:    maximize log(D(fake))  (make the fake look real)
```

This project follows the widely used DCGAN recipe:

- **Optimizer**: Adam with learning rate `2e-4` and `beta1 = 0.5` keeps training steady.
- **Label smoothing**: we use `0.9` instead of `1.0` for “real” labels to prevent the discriminator from becoming overconfident.
- **Input noise**: the generator always begins with random noise (the “creative spark”), so it explores different faces.

---

## 3. What You Need To Run Training

- **CelebA dataset** (aligned & cropped version recommended). Put the image files inside `data/small` or `data/large`.
- **Python packages**: `torch`, `torchvision`, `tqdm`, and `pillow`.

Install dependencies from a terminal:

```cmd
pip install torch torchvision tqdm pillow
```

---

## 4. Running The Training Script

Launch training with:

```cmd
python main.py
```

What happens next:

- The script loads CelebA faces, rescales them to 64×64, and feeds them in batches.
- The generator and discriminator play their contest for the number of epochs you choose (default 50).
- After each epoch you get:
  - **Sample grid** in `outputs/epoch_XXX.png`: shows how the fake faces are progressing.
  - **Checkpoint** in `models/models_XXX.pth`: lets you resume training or generate faces later.
  - **Log entry** in `training_log.csv`: average discriminator and generator loss for that epoch.

Training is rarely smooth—loss values can bounce around. Focus on the image grids to judge progress; they tell the real story.

---

## 5. Understanding The Outputs

- **PNG grids**: these use the same fixed noise each epoch, so you can clearly see improvements (sharper features, more diverse faces, fewer artifacts).
- **Model checkpoints**: load one later to generate new faces without retraining.
- **CSV log**: helpful for plotting loss curves or spotting instability.

Example of sampling after training finishes:

```python
import torch
from torchvision.utils import save_image
from src._generator import Generator

checkpoint = torch.load("models/models_050.pth", map_location="cpu")
netG = Generator(nz=100, ngf=64)
netG.load_state_dict(checkpoint['netG'])
netG.eval()

noise = torch.randn(16, 100, 1, 1)
with torch.no_grad():
        fakes = netG(noise)
save_image((fakes + 1) / 2, "sample_grid.png", nrow=4)
```

---

## 6. Tuning Experiments (No Code Changes Needed)

- **Epoch count**: more epochs mean sharper faces but longer training time.
- **Batch size**: use the largest your GPU allows—larger batches often stabilize GANs.
- **Latent size (`nz`)**: increasing it gives the generator more imagination, though training might slow.
- **Learning rate**: if the model oscillates or collapses, try lowering `lr` from `2e-4` to `1e-4`.
- **Label smoothing**: experiment with values like `real=0.95`, `fake=0.05` to fight overconfidence.

Remember that GANs are sensitive. Small tweaks can make training more stable—or less. Always watch the image grids to know if a change helps.

---

## 7. Common Hiccups

- **Faces don’t improve**: double-check your dataset path and make sure images are loading. Also confirm the generator and discriminator losses are actually changing.
- **Checkerboard artifacts**: normal for DCGANs. Try lowering the learning rate or training longer.
- **Mode collapse (all faces look identical)**: add label noise, lower the learning rate, or train the discriminator slightly less often.
- **CUDA out of memory**: lower the batch size or run on CPU (training will be slower but still functional).

---

## 8. Want To Learn More?

- DCGAN paper — *Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks* (Radford et al., 2015)
- Goodfellow’s original GAN paper — *Generative Adversarial Networks* (2014)
- PyTorch DCGAN tutorial — https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html

Enjoy exploring! GANs are equal parts science and art—experiment freely.
