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
        self.g_steps_per_d_step = g_steps_per_d_step  # new

    def train_generator_one_epoch(self):
        # Clear generator gradients
        self.g_optimizer.zero_grad()

        # Generate fake images
        latent = torch.randn(self.batch_size, self.generator.latent_dim, 1, 1).to(self.device)  # random noice
        fake_images = self.generator(latent).to(self.device)  # fake images generated

        # Try to fool the discriminator
        preds = self.discriminator(fake_images).to(self.device)  # getting the predictions of discriminator for fake images
        targets = torch.ones_like(preds).to(self.device)  # setting 1 as targets so the discriminator can be fooled
        loss = F.binary_cross_entropy(preds, targets)  # comparing

        # Update generator weights
        loss.backward()
        self.g_optimizer.step()

        return loss.item()

    def train_discriminator_one_epoch(self, real_images_batch):

        # Clear discriminator gradients
        self.d_optimizer.zero_grad()

        # Pass real images through discriminator
        real_preds = self.discriminator(real_images_batch).to(self.device)  # real images
        real_targets = torch.ones_like(real_preds).to(self.device)  # setting targets as 1
        real_loss = F.binary_cross_entropy(real_preds, real_targets)  # getting the loss

        # Generate fake images
        latent = torch.randn(self.batch_size, self.generator.latent_dim, 1, 1).to(self.device)  # generating the random noices for input image
        fake_images = self.generator(latent).to(self.device)  # getting the fake images

        # Pass fake images through discriminator
        fake_preds = self.discriminator(fake_images).to(self.device)  # getting the predictions for fake images
        fake_targets = torch.zeros_like(fake_preds).to(self.device)  # setting 0 as target for fake images
        fake_loss = F.binary_cross_entropy(fake_preds, fake_targets)  # Comparing the two scores through loss

        # Update discriminator weights
        loss = real_loss + fake_loss
        loss.backward()
        self.d_optimizer.step()
        return loss.item()

    def train(self, epochs):
        g_losses = []
        d_losses = []
        for epoch in range(epochs):
            for real_images, _ in tqdm(self.dataloader):
                real_images = real_images.to(self.device)

                # Train D once
                d_loss = self.train_discriminator_one_epoch(real_images)

                # Train G multiple times
                g_loss = 0
                for _ in range(self.g_steps_per_d_step):
                    g_loss += self.train_generator_one_epoch()
                g_loss /= self.g_steps_per_d_step  # average G loss for cleaner reporting

            generate_and_save_images(generator=self.generator, latent_dim=self.generator.latent_dim, device=self.device,
                                         save_dir=f'generated_images_epoch{epoch+1}')

            d_losses.append(d_loss)
            g_losses.append(g_loss)

            print(f'Epoch [{epoch+1}/{epochs}], loss_g: {g_loss:.4f}, loss_d: {d_loss:.4f}')

        return g_losses, d_losses