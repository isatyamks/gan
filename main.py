from src._train import train_dcgan

if __name__ == "__main__":
    train_dcgan(
        data_root="data\\small",   # <-- your CelebA local path
        out_dir="outputs",
        model_dir = "models",
        image_size=64,
        batch_size=128,
        nz=100,
        ngf=64,
        ndf=64,
        epochs=50,
        lr=2e-4,
        beta1=0.5
    )
