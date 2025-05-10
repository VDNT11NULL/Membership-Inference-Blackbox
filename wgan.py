import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import os
import random
import numpy as np
from tqdm import tqdm
import torchvision

class Generator(nn.Module):
    def __init__(self, z_dim, img_channels, features_g):
        super().__init__()
        self.gen = nn.Sequential(
            # Input: N x z_dim x 1 x 1
            self._block(z_dim, features_g*16, 4, 1, 0),  # N x f_g*16 x 4 x 4
            self._block(features_g*16, features_g*8, 4, 2, 1),  # N x f_g*8 x 8 x 8
            self._block(features_g*8, features_g*4, 4, 2, 1),  # N x f_g*4 x 16 x 16
            self._block(features_g*4, features_g*2, 4, 2, 1),  # N x f_g*2 x 32 x 32
            nn.ConvTranspose2d(features_g*2, img_channels, 4, 2, 1),  # N x img_channels x 64 x 64
            nn.Tanh()
        )
    
    def _block(self, in_ch, out_ch, kernel_size, stride, padding):
        return nn.Sequential(
            nn.ConvTranspose2d(in_ch, out_ch, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        # Ensure input has correct shape [batch_size, z_dim, 1, 1]
        if x.dim() == 2:
            x = x.unsqueeze(-1).unsqueeze(-1)
        return self.gen(x)

class Discriminator(nn.Module):
    def __init__(self, img_channels, features_dim):
        super().__init__()
        self.disc = nn.Sequential(
            # Input: N x img_channels x 64 x 64
            nn.Conv2d(img_channels, features_dim, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            self._block(features_dim, features_dim*2, 4, 2, 1),  # 32 x 32
            self._block(features_dim*2, features_dim*4, 4, 2, 1),  # 16 x 16
            self._block(features_dim*4, features_dim*8, 4, 2, 1),  # 8 x 8
            self._block(features_dim*8, features_dim*16, 4, 2, 1),  # 4 x 4
            nn.Conv2d(features_dim*16, 1, 4, 2, 1)  # 1 x 1
        )
    
    def _block(self, in_ch, out_ch, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding, bias=False),
            nn.LeakyReLU(0.2)
        )
    
    def forward(self, x):
        # Ensure input has correct shape [batch_size, channels, height, width]
        if x.dim() == 3:
            x = x.unsqueeze(0)
        return self.disc(x)

def init_weights(model):
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.BatchNorm2d, nn.ConvTranspose2d)):
            nn.init.normal_(m.weight.data, 0.0, 0.02)

def train_wgan(
    generator,
    discriminator,
    dataloader,
    num_epochs,
    z_dim,
    lr=0.0002,
    beta1=0.5,
    n_critic=5,  # Number of D updates per G update
    clip_value=0.01,  # Weight clipping value
    device="cuda" if torch.cuda.is_available() else "cpu"
):
    # Initialize optimizers
    g_optimizer = optim.Adam(generator.parameters(), lr=lr, betas=(beta1, 0.999))
    d_optimizer = optim.Adam(discriminator.parameters(), lr=lr, betas=(beta1, 0.999))
    
    # Initialize weights
    generator.apply(init_weights)
    discriminator.apply(init_weights)
    
    # Move models to device
    generator.to(device)
    discriminator.to(device)
    
    # Set models to training mode
    generator.train()
    discriminator.train()
    
    for epoch in range(num_epochs):
        for batch_idx, (real_images, _) in enumerate(dataloader):
            try:
                batch_size = real_images.size(0)
                real_images = real_images.to(device)
                
                # Ensure real images are properly normalized
                if real_images.max() > 1.0 or real_images.min() < -1.0:
                    real_images = torch.clamp(real_images, -1.0, 1.0)
                
                # Train Discriminator
                d_optimizer.zero_grad()
                
                # Generate fake images
                z = torch.randn(batch_size, z_dim, 1, 1).to(device)
                fake_images = generator(z)
                
                # Ensure fake images are properly normalized
                fake_images = torch.clamp(fake_images, -1.0, 1.0)
                
                # Compute Wasserstein distance
                d_real = discriminator(real_images)
                d_fake = discriminator(fake_images.detach())  # Detach to avoid computing gradients for G
                
                # WGAN loss
                d_loss = -(torch.mean(d_real) - torch.mean(d_fake))
                d_loss.backward()
                d_optimizer.step()
                
                # Clip weights of discriminator
                for p in discriminator.parameters():
                    p.data.clamp_(-clip_value, clip_value)
                
                # Train Generator every n_critic iterations
                if batch_idx % n_critic == 0:
                    g_optimizer.zero_grad()
                    
                    # Generate new fake images
                    fake_images = generator(z)
                    d_fake = discriminator(fake_images)
                    
                    # WGAN loss for generator
                    g_loss = -torch.mean(d_fake)
                    g_loss.backward()
                    g_optimizer.step()
                
                if batch_idx % 100 == 0:
                    print(f"Epoch [{epoch}/{num_epochs}] Batch [{batch_idx}/{len(dataloader)}] "
                          f"D_loss: {d_loss.item():.4f} G_loss: {g_loss.item():.4f}")
            
            except RuntimeError as e:
                print(f"Error in batch {batch_idx}: {str(e)}")
                print(f"Real images shape: {real_images.shape}")
                print(f"Fake images shape: {fake_images.shape if 'fake_images' in locals() else 'Not created yet'}")
                raise e

def train_reconstruction_model(gen, dataloader, dataset, NUM_EPOCHS=200, num_samples=None, save_dir='../Submission/checkpoints'):
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs('../Submission/Real_Ones', exist_ok=True)
    os.makedirs('../Submission/Gen_Fakes', exist_ok=True)
    
    if num_samples:
        indices = torch.randperm(len(dataset))[:num_samples]
        dataset = torch.utils.data.Subset(dataset, indices)
        dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    try:
        # Initialize generator
        gen = Generator(Z_DIM, IMG_CHANNELS, FEATURES_G).to(DEVICE)
        init_weights(gen)
        
        # Optimizer for generator
        opt_gen = optim.Adam(gen.parameters(), lr=LEARNING_RATE_G, betas=BETAS)
        
        # Learning rate scheduler
        scheduler_g = optim.lr_scheduler.ReduceLROnPlateau(opt_gen, mode='min', factor=0.5, patience=1000, verbose=True)
        
        # Augmentation policy
        augment_policy = 'brightness,contrast,saturation'
        
        # MSE loss for reconstruction
        mse_loss = nn.MSELoss()
        
        # Fixed noise for visualization
        fixed_noise = torch.randn(32, Z_DIM, 1, 1).to(DEVICE)
        
        loop = tqdm(range(NUM_EPOCHS), desc='Training Reconstruction Model', colour='green')
        for epoch in loop:
            total_g_loss = 0
            
            for batch_idx, (noise, real_images) in enumerate(dataloader):
                noise = noise.to(DEVICE)
                real_images = real_images.to(DEVICE)
                batch_size = real_images.shape[0]
                
                # Apply differential augmentation
                real_augmented = diff_augment(real_images, policy=augment_policy)
                
                # Generate fake images
                fake_images = gen(noise)
                fake_augmented = diff_augment(fake_images, policy=augment_policy)
                
                # Reconstruction loss
                g_loss = mse_loss(fake_augmented, real_augmented)
                
                # Optional: Add perceptual loss using discriminator features
                # if disc is not None:
                #     perceptual_loss = compute_lpips_loss(disc, real_augmented, fake_augmented)
                #     g_loss = g_loss + 0.1 * perceptual_loss
                
                opt_gen.zero_grad()
                g_loss.backward()
                opt_gen.step()
                
                total_g_loss += g_loss.item()
            
            # Update learning rate
            avg_g_loss = total_g_loss / len(dataloader)
            scheduler_g.step(avg_g_loss)
            
            # Save images periodically
            if epoch % 15 == 0 or epoch == NUM_EPOCHS - 1:
                with torch.no_grad():
                    # Generate using fixed noise for consistency
                    fake = gen(fixed_noise)
                    img_grid_fake = torchvision.utils.make_grid(fake[:32], normalize=True)
                    
                    # Get batch of real images
                    real_batch = next(iter(dataloader))[1][:32].to(DEVICE)
                    img_grid_real = torchvision.utils.make_grid(real_batch, normalize=True)
                    
                    # Save images
                    torchvision.utils.save_image(img_grid_real, f'../Submission/Real_Ones/reconstruction_real_epoch_{epoch}.png')
                    torchvision.utils.save_image(img_grid_fake, f'../Submission/Gen_Fakes/reconstruction_fake_epoch_{epoch}.png')
            
            # Save model checkpoints
            if epoch % SAVE_INTERVAL == 0 or epoch == NUM_EPOCHS - 1:
                torch.save({
                    'epoch': epoch,
                    'gen_state_dict': gen.state_dict(),
                    'opt_gen': opt_gen.state_dict(),
                }, f'{save_dir}/reconstruction_checkpoint_epoch_{epoch}.pt')
            
            loop.set_description(f'Epoch: {epoch} | G Recon Loss: {avg_g_loss:.4f}')
        
    except Exception as e:
        print(f"An error occurred during training: {str(e)}")
        raise e
    finally:
        # Clean up
        if 'dataloader' in locals():
            del dataloader
        if 'gen' in locals():
            del gen
        torch.cuda.empty_cache()

# Main function for reconstruction training
def main_reconstruction():
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    np.random.seed(42)
    random.seed(42)
    
    # Set paths
    noise_dir = '../Submission/NEW_QUery_generated_dataset/noise_vectors'
    qgen_images_dir = '../Submission/NEW_QUery_generated_dataset/images'
    
    # Setup data
    train_ds, train_loader = setup_data(noise_dir, qgen_images_dir)
    
    # Create generator
    gen = Generator(Z_DIM, IMG_CHANNELS, FEATURES_G).to(DEVICE)
    
    # Run training
    print("Starting reconstruction training...")
    train_reconstruction_model(
        gen, 
        train_loader, 
        train_ds, 
        NUM_EPOCHS=200, 
        num_samples=None,  # Use all samples
        save_dir='../Submission/checkpoints'
    )
    print("Reconstruction training complete!")

if __name__ == "__main__":
    main_reconstruction() 