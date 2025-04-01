import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

## MAKE SURE TO RUN THE SCRIPT in the same directory as the model.py file (../GAN-Leaks/gan_models/dcgan/dcgan_identity0_20k/identity0_20k)
from model import DCGAN

tf.compat.v1.disable_eager_execution()

CHECKPOINT_DIR = "../GAN-Leaks/gan_models/dcgan/dcgan_identity0_20k/identity0_20k"  
OUTPUT_DIR = "generated_image_dataset" 
NUM_SAMPLES = 100  

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

with tf.compat.v1.Session() as sess:
    dcgan = DCGAN(sess, batch_size=1, output_height=64, output_width=64)
    
    print(f"Loading checkpoint from: {CHECKPOINT_DIR}")
    could_load, _ = dcgan.load(CHECKPOINT_DIR)
    if not could_load:
        raise Exception(f"Failed to load checkpoint from {CHECKPOINT_DIR}. Make sure the directory contains valid checkpoint files.")

    z_vectors = []
    generated_images = []

    for i in range(NUM_SAMPLES):

        z = np.random.normal(0, 1, size=[1, dcgan.z_dim])
        
        sample = sess.run(dcgan.sampler, feed_dict={dcgan.z: z})
        
        z_vectors.append(z)
        generated_images.append(sample[0])
        
        img_path = os.path.join(OUTPUT_DIR, f'generated_image_{i}.png')
        save_images(sample, [1, 1], img_path)

    np.save(os.path.join(OUTPUT_DIR, 'z_vectors.npy'), np.array(z_vectors))
    np.save(os.path.join(OUTPUT_DIR, 'generated_images.npy'), np.array(generated_images))
    print(f"Dataset of {NUM_SAMPLES} (z, G(z)) pairs saved to {OUTPUT_DIR}")
