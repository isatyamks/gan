import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import utils
from tqdm import tqdm

from src._generator import Generator
from src._discriminator import Discriminator
from src._dataset import get_celeba_dataloader
from src._utils import weights_init, init_csv

def train_dcgan(data_root, out_dir,model_dir, image_size=64, batch_size=128,
                nz=100, ngf=64, ndf=64, epochs=50, lr=2e-4, beta1=0.5):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(out_dir, exist_ok=True)

    # Data loader
    loader = get_celeba_dataloader(data_root, image_size, batch_size)

    # Models
    netG = Generator(nz, ngf).to(device)
    netD = Discriminator(ndf).to(device)
    netG.apply(weights_init)
    netD.apply(weights_init)

    # Loss + optimizer
    criterion = nn.BCELoss()
    optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))
    fixed_noise = torch.randn(64, nz, 1, 1, device=device)

    csv_file, csv_writer = init_csv("./")
    real_label = 0.9
    fake_label = 0.0

    for epoch in range(epochs):
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{epochs}")
        errD_sum, errG_sum, batches = 0, 0, 0

        for images, _ in pbar:
            images = images.to(device)
            b_size = images.size(0)

            # Train Discriminator
            netD.zero_grad()
            label = torch.full((b_size,), real_label, device=device)
            output = netD(images)
            errD_real = criterion(output, label)
            errD_real.backward()

            noise = torch.randn(b_size, nz, 1, 1, device=device)
            fake = netG(noise)
            label.fill_(fake_label)
            output_fake = netD(fake.detach())
            errD_fake = criterion(output_fake, label)
            errD_fake.backward()
            optimizerD.step()

            errD = errD_real + errD_fake

            # Train Generator
            netG.zero_grad()
            label.fill_(real_label)
            output2 = netD(fake)
            errG = criterion(output2, label)
            errG.backward()
            optimizerG.step()

            errD_sum += errD.item()
            errG_sum += errG.item()
            batches += 1
            pbar.set_postfix({'errD': errD.item(), 'errG': errG.item()})

        # Epoch summary
        avg_errD = errD_sum / batches
        avg_errG = errG_sum / batches
        csv_writer.writerow([epoch+1, avg_errD, avg_errG, batch_size, lr, beta1, ngf, ndf, nz])
        csv_file.flush()

        with torch.no_grad():
            sample = netG(fixed_noise).detach().cpu()
        utils.save_image((sample + 1) / 2.0, os.path.join(out_dir, f"epoch_{epoch+1:03d}.png"), nrow=8)

        torch.save({'netG': netG.state_dict(), 'netD': netD.state_dict()},
                   os.path.join(model_dir, f"models_{epoch+1:03d}.pth"))

    csv_file.close()
    print("training done....")
