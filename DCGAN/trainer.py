import torch
import torch.nn.functional as F
from tqdm import tqdm

from DCGAN.image_saver import generate_and_save_images


class Trainer:
    def __init__(self, generator, discriminator, g_optimizer, d_optimizer, dataloader, batch_size, device, label_smoothing=0.1, g_steps_per_d_step=1):
        self.generator = generator
        self.discriminator = discriminator
        self.g_optimizer = g_optimizer
        self.d_optimizer = d_optimizer
        self.batch_size = batch_size
        self.device = device
        self.dataloader = dataloader
        self.label_smoothing = label_smoothing
        self.g_steps_per_d_step = g_steps_per_d_step  # new

    def train_generator_one_epoch(self, fake_images):
        self.generator.train()

        self.g_optimizer.zero_grad()
        preds = self.discriminator(fake_images)
        # Label smoothing for real-like labels
        real_like_labels = torch.full_like(preds, 1.0 - self.label_smoothing).to(self.device)
        loss = F.binary_cross_entropy(preds, real_like_labels)
        loss.backward()
        self.g_optimizer.step()
        return loss.item()

    def train_discriminator_one_epoch(self, real_images_batch, fake_images):
        self.discriminator.train()

        preds_fake = self.discriminator(fake_images.detach())
        fake_labels = torch.zeros_like(preds_fake).to(self.device)

        fake_loss = F.binary_cross_entropy(preds_fake, fake_labels)

        preds_real = self.discriminator(real_images_batch)
        smoothed_real_labels = torch.full_like(preds_real, 1.0 - self.label_smoothing).to(self.device)
        real_loss = F.binary_cross_entropy(preds_real, smoothed_real_labels)

        self.d_optimizer.zero_grad()
        loss = (fake_loss + real_loss) / 2
        loss.backward()
        self.d_optimizer.step()

        return loss.item()

    def train(self, epochs):
        g_losses = []
        d_losses = []
        for epoch in range(epochs):
            for real_images, _ in tqdm(self.dataloader):
                real_images = real_images.to(self.device)
                fake_latent = torch.randn((self.batch_size, self.generator.latent_dim, 1, 1)).to(self.device)
                fake_images = self.generator(fake_latent)

                # Train D once
                d_loss = self.train_discriminator_one_epoch(real_images, fake_images)

                # Train G multiple times
                g_loss = 0
                for _ in range(self.g_steps_per_d_step):
                    g_loss += self.train_generator_one_epoch(fake_images)
                g_loss /= self.g_steps_per_d_step  # average G loss for cleaner reporting

            if epoch % 5 == 0:
                generate_and_save_images(generator=self.generator, latent_dim=self.generator.latent_dim, device=self.device,
                                         save_dir=f'generated_images_epoch{epoch+1}')

            d_losses.append(d_loss)
            g_losses.append(g_loss)

            print(f'Epoch [{epoch+1}/{epochs}], loss_g: {g_loss:.4f}, loss_d: {d_loss:.4f}')

        return g_losses, d_losses