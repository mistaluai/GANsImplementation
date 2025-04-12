import os
import torch
import torchvision.utils as vutils

def generate_and_save_images(generator, latent_dim, device, save_dir="generated_images"):
    os.makedirs(save_dir, exist_ok=True)

    generator.eval()

    with torch.no_grad():
        fake_images = []
        for _ in range(10):
            z = torch.randn(1, latent_dim, 1, 1).to(device)
            generated_image = generator(z)
            generated_image = (generated_image + 1) / 2.0
            fake_images.append(generated_image)

    for i in range(10):
        img = fake_images[i]
        save_path = os.path.join(save_dir, f"fake_image_{i+1}.png")
        vutils.save_image(img, save_path)