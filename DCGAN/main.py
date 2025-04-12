import torchvision.datasets as datasets
from torchvision.transforms import v2
import torch

from image_saver import generate_and_save_images
from trainer import Trainer
from model import Generator, Discriminator, initialize_weights

batch_size = 128
device = 'cuda' if torch.cuda.is_available() else 'mps'

transforms = v2.Compose([
    v2.Resize((64, 64)),
    v2.ToImage(),
    v2.ToDtype(torch.float32),
    v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])
data = datasets.CIFAR10(download=True, transform=transforms, root='./data')
train_loader = torch.utils.data.DataLoader(data, batch_size=batch_size)


generator = Generator().to(device)
discriminator = Discriminator().to(device)

initialize_weights(generator)
initialize_weights(discriminator)

optim_g = torch.optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
optim_d = torch.optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

trainer = Trainer(generator, discriminator, optim_g, optim_d, train_loader, batch_size, device)

epochs = 1
trainer.train(epochs=epochs)

generate_and_save_images(generator=generator, latent_dim=generator.latent_dim, device=device, save_dir='generated_images')