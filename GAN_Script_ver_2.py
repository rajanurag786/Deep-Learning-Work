# -*- coding: utf-8 -*-
"""
Created on Wed Oct 16 11:03:29 2024

@author: ge87fam
"""

import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import json
import os
from PIL import Image
import time

tf.keras.backend.set_floatx('float32')

def wasserstein_loss(y_true, y_pred):
    return tf.reduce_mean(y_true * y_pred)

def gradient_penalty(discriminator, real_images, fake_images, labels):
    alpha = tf.random.uniform([tf.shape(real_images)[0], 1, 1, 1], 0.0, 1.0, dtype=tf.float32)
    interpolated = alpha * real_images + (1 - alpha) * fake_images
    
    with tf.GradientTape() as tape:
        tape.watch(interpolated)
        predictions = discriminator([interpolated, labels])
    
    gradients = tape.gradient(predictions, interpolated)
    slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), axis=[1, 2, 3]))
    return tf.reduce_mean((slopes - 1.0) ** 2)

# Define the Generator
def build_generator(latent_dim, num_points):
    noise_input = keras.Input(shape=(latent_dim,), dtype=tf.float32)
    label_input = keras.Input(shape=(num_points*2,), dtype=tf.float32)
    
    x = keras.layers.Concatenate()([noise_input, label_input])
    x = keras.layers.Dense(8 * 8 * 256, kernel_initializer='he_normal')(x)
    x = keras.layers.LeakyReLU(alpha=0.2)(x)
    x = keras.layers.Reshape((8, 8, 256))(x)
    
    x = keras.layers.Conv2DTranspose(128, 4, strides=2, padding='same', kernel_initializer='he_normal')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.LeakyReLU(alpha=0.2)(x)
    
    x = keras.layers.Conv2DTranspose(64, 4, strides=2, padding='same', kernel_initializer='he_normal')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.LeakyReLU(alpha=0.2)(x)
    
    x = keras.layers.Conv2DTranspose(32, 4, strides=2, padding='same', kernel_initializer='he_normal')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.LeakyReLU(alpha=0.2)(x)
    
    x = keras.layers.Conv2DTranspose(16, 4, strides=2, padding='same')(x)  # 128x128
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.LeakyReLU(alpha=0.2)(x)

    x = keras.layers.Conv2DTranspose(8, 4, strides=2, padding='same')(x)  # 256x256
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.LeakyReLU(alpha=0.2)(x)
    
    x = keras.layers.Conv2DTranspose(1, 4, strides=2, padding='same', activation='leaky_relu', kernel_initializer='he_normal')(x)
    
    return keras.Model([noise_input, label_input], x)


# Define the Discriminator
def build_discriminator(img_shape, num_points):
    img_input = keras.Input(shape=img_shape, dtype=tf.float32)
    label_input = keras.Input(shape=(num_points*2,), dtype=tf.float32)
    
    x = keras.layers.Conv2D(64, 4, strides=2, padding='same', kernel_initializer='he_normal')(img_input)
    x = keras.layers.LeakyReLU(alpha=0.2)(x)
    
    x = keras.layers.Conv2D(128, 4, strides=2, padding='same', kernel_initializer='he_normal')(x)
    x = keras.layers.LayerNormalization()(x)
    x = keras.layers.LeakyReLU(alpha=0.2)(x)
    
    x = keras.layers.Conv2D(256, 4, strides=2, padding='same', kernel_initializer='he_normal')(x)
    x = keras.layers.LayerNormalization()(x)
    x = keras.layers.LeakyReLU(alpha=0.2)(x)
    
    x = keras.layers.Flatten()(x)
    
    label_embedding = keras.layers.Dense(512, activation='leaky_relu', kernel_initializer='he_normal')(label_input)
    x = keras.layers.Concatenate()([x, label_embedding])
    
    x = keras.layers.Dense(512, activation='leaky_relu', kernel_initializer='he_normal')(x)
    x = keras.layers.Dense(1, kernel_initializer='he_normal')(x)
    
    return keras.Model([img_input, label_input], x)

# Define the CGAN
class TrussCGAN(keras.Model):
    def __init__(self, generator, discriminator):
        super(TrussCGAN, self).__init__()
        self.generator = generator
        self.discriminator = discriminator

    def compile(self, g_optimizer, d_optimizer, gp_weight=5.0):
        super(TrussCGAN, self).compile()
        self.g_optimizer = g_optimizer
        self.d_optimizer = d_optimizer
        self.gp_weight = tf.cast(gp_weight, dtype=tf.float32)

    @tf.function
    def train_step(self, data):
        real_images, labels = data
        batch_size = tf.shape(real_images)[0]
        latent_dim = self.generator.input_shape[0][1]

        # Train the discriminator
        for _ in range(2):  # Reduced number of discriminator updates
            random_latent_vectors = tf.random.normal(shape=(batch_size, latent_dim), dtype=tf.float32)
            with tf.GradientTape() as tape:
                fake_images = self.generator([random_latent_vectors, labels])
                fake_predictions = self.discriminator([fake_images, labels])
                real_predictions = self.discriminator([real_images, labels])

                d_loss_real = tf.reduce_mean(real_predictions)
                d_loss_fake = tf.reduce_mean(fake_predictions)
                d_loss = d_loss_fake - d_loss_real

                # Gradient penalty
                gp = self.gradient_penalty(real_images, fake_images, labels)
                d_loss += self.gp_weight * gp

            d_gradients = tape.gradient(d_loss, self.discriminator.trainable_variables)
            d_gradients, _ = tf.clip_by_global_norm(d_gradients, 1.0)  # Gradient clipping
            self.d_optimizer.apply_gradients(zip(d_gradients, self.discriminator.trainable_variables))

        # Train the generator
        random_latent_vectors = tf.random.normal(shape=(batch_size, latent_dim), dtype=tf.float32)
        with tf.GradientTape() as tape:
            fake_images = self.generator([random_latent_vectors, labels])
            fake_predictions = self.discriminator([fake_images, labels])
            g_loss = -tf.reduce_mean(fake_predictions)

        g_gradients = tape.gradient(g_loss, self.generator.trainable_variables)
        g_gradients, _ = tf.clip_by_global_norm(g_gradients, 1.0)  # Gradient clipping
        self.g_optimizer.apply_gradients(zip(g_gradients, self.generator.trainable_variables))

        return {
            "d_loss": d_loss,
            "g_loss": g_loss,
            "gp": gp,
            "d_loss_real": d_loss_real,
            "d_loss_fake": d_loss_fake
        }
    
    def gradient_penalty(self, real_images, fake_images, labels):
        alpha = tf.random.uniform([tf.shape(real_images)[0], 1, 1, 1], 0.0, 1.0)
        interpolated = alpha * real_images + (1 - alpha) * fake_images
        
        with tf.GradientTape() as tape:
            tape.watch(interpolated)
            predictions = self.discriminator([interpolated, labels])
        
        gradients = tape.gradient(predictions, interpolated)
        slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), axis=[1, 2, 3]))
        return tf.reduce_mean((slopes - 1.0) ** 2)

           

# Data preparation function
def prepare_dataset(json_file, image_data_path, batch_size):
    # Load JSON data
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    # Extract frequencies and displacements
    frequencies = np.array([sample['frequencies'] for sample in data], dtype=np.float32)
    displacements = np.array([sample['displacements_x'] for sample in data], dtype=np.float32)
    
    # Normalize frequencies and displacements
    frequencies = (frequencies - np.min(frequencies)) / (np.max(frequencies) - np.min(frequencies))
    displacements = (displacements - np.min(displacements)) / (np.max(displacements) - np.min(displacements))
    
    # Combine frequencies and displacements
    labels = np.concatenate((frequencies, displacements), axis=1)
    
    # Load image data
    if image_data_path.endswith('.npy'):
        # If image data is stored as a single numpy array
        images = np.load(image_data_path).astype(np.float32)
    else:
        # If image data is stored as individual files
        image_files = sorted(os.listdir(image_data_path))
        images = []
        for img_file in image_files:
            img_path = os.path.join(image_data_path, img_file)
            img = Image.open(img_path).convert('L')  # Convert to grayscale
            img_array = np.array(img, dtype=np.float32) / 255.0  # Normalize to [0, 1]
            images.append(img_array)
        images = np.array(images, dtype=np.float32)
    
    # Ensure images are in the correct shape (num_samples, 512, 512, 1)
    images = images.reshape((-1, 512, 512, 1))
    
    # Create TensorFlow dataset
    dataset = tf.data.Dataset.from_tensor_slices((images, labels))
    return dataset.shuffle(buffer_size=1000).batch(batch_size)


def generate_and_save_images(model, epoch, test_input, test_labels, save_dir):
    
    save_dir = r"H:\\Deep Learning Models\\GAN Generated images"
    os.makedirs(save_dir, exist_ok=True)
    predictions = model.generator([test_input, test_labels])
    
    fig = plt.figure(figsize=(10, 5))
    
    # for i in range(predictions.shape[0]):
    #     plt.subplot(2, 2, i+1)
    #     plt.imshow(predictions[i, :, :, 0], cmap='gray')
    #     plt.title(f"F: {test_labels[i, 0]:.2f}, D: {test_labels[i, 1]:.2f}")
    #     plt.axis('off')
    for i in range(predictions.shape[0]):
        plt.imshow(predictions[i, :, :, 0], cmap='gray')
        plt.title(f"F: {test_labels[i, 0]:.2f}, D: {test_labels[i, 1]:.2f}")
        plt.axis('off')
    
    # plt.tight_layout()
    # save_path = os.path.join(save_dir, f'truss_unit_epoch_{epoch:04d}.png')
    # plt.savefig(save_path)
    # print(fig)
    # plt.close()
    individual_save_path = os.path.join(save_dir, f'truss_unit_epoch_{epoch:04d}_image_{i+1}.png')
    plt.savefig(individual_save_path)
    print(fig)
    plt.close()
    
    
# Main training loop
def train_cgan(json_file, image_data_path, epochs=2, batch_size=16, latent_dim=600):
    # Load a single sample to get the number of points
    with open(json_file, 'r') as f:
        data = json.load(f)
    num_points = len(data[0]['frequencies'])

    img_shape = (512, 512, 1)

    generator = build_generator(latent_dim, num_points)
    discriminator = build_discriminator(img_shape, num_points)

    cgan = TrussCGAN(generator, discriminator)
    cgan.compile(
        g_optimizer=keras.optimizers.Adam(learning_rate=0.0001, beta_1=0.5, beta_2=0.9),
        d_optimizer=keras.optimizers.Adam(learning_rate=0.0001, beta_1=0.5, beta_2=0.9, clipvalue=1.0)
    )

    dataset = prepare_dataset(json_file, image_data_path, batch_size)

    # Prepare fixed input for image generation
    num_examples_to_generate = 4
    test_input = tf.random.normal([num_examples_to_generate, latent_dim])
    test_labels = tf.random.normal([num_examples_to_generate, num_points * 2])

    d_losses_over_time = []
    g_losses_over_time = []

    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")
        d_losses = []
        g_losses = []
        gp_values = []
        d_losses_real = []
        d_losses_fake = []

        start_time = time.time()  # Start time for the epoch
        total_steps = len(dataset)

        for step, (batch_images, batch_labels) in enumerate(dataset):
            step_start_time = time.time()  # Start time for the step

            # Perform a training step
            metrics = cgan.train_step((batch_images, batch_labels))
            d_losses.append(float(metrics['d_loss']))
            g_losses.append(float(metrics['g_loss']))
            gp_values.append(float(metrics['gp']))
            d_losses_real.append(float(metrics['d_loss_real']))
            d_losses_fake.append(float(metrics['d_loss_fake']))

            # Calculate and display step progress
            step_time = time.time() - step_start_time
            steps_remaining = total_steps - (step + 1)
            eta = steps_remaining * step_time
            print(
                f"Step {step + 1}/{total_steps} | "
                f"Step Time: {step_time:.2f}s | "
                f"Epoch ETA: {int(eta // 60)}m {int(eta % 60)}s",
                end='\n'
            )

        # Epoch time and statistics
        epoch_time = time.time() - start_time
        d_losses_over_time.append(np.mean(d_losses))
        g_losses_over_time.append(np.mean(g_losses))

        print(f"\nEpoch Time: {int(epoch_time // 60)}m {int(epoch_time % 60)}s")
        print(f"D Loss: {np.mean(d_losses):.4f}, G Loss: {np.mean(g_losses):.4f}")
        print(f"GP: {np.mean(gp_values):.4f}")
        print(f"D Loss Real: {np.mean(d_losses_real):.4f}, D Loss Fake: {np.mean(d_losses_fake):.4f}")

        # Generate and save sample images every 5 epochs
        if (epoch + 1) % 5 == 0:
            generate_and_save_images(cgan, epoch + 1, test_input, test_labels, save_dir=True)

        # Early stopping if losses become NaN
        if np.isnan(np.mean(d_losses)) or np.isnan(np.mean(g_losses)):
            print("NaN loss detected. Stopping training.")
            break

    return cgan, d_losses_over_time, g_losses_over_time

# Function to generate new truss units
def generate_truss_units(cgan, num_samples, desired_freqs, desired_disp, latent_dim=600):
    # Ensure desired_freqs is a list
    if not isinstance(desired_freqs, list):
        desired_freqs = [desired_freqs]
        
    for freqs in desired_freqs:
        
        random_latent_vectors = tf.random.normal(shape=(num_samples, latent_dim))
        desired_labels = np.array([[desired_freqs, desired_disp] for _ in range(num_samples)])
        expanded_labels = np.pad(desired_labels, ((0, 0), (0, 198)), 'constant')
        generated_images = cgan.generator.predict([random_latent_vectors, expanded_labels])
        
        # Post-process to get binary images
        binary_images = (generated_images > 0.5).astype(np.float32)
    
    return binary_images

# Main execution
if __name__ == "__main__":
    json_file = r"H:\\Deep Learning Models\\all_results.json"  # JSON file path
    image_data_path = r"H:\\Deep Learning Models\\image_samples_2500"  # Your image data path (either .npy file or directory)

    # Train the model
    trained_cgan, d_losses, g_losses = train_cgan(json_file, image_data_path, epochs=2)
    
    # Plot loss curves
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(d_losses) + 1), d_losses, label='Discriminator Loss')
    plt.plot(range(1, len(g_losses) + 1), g_losses, label='Generator Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss Curves for GAN Training')
    plt.legend()
    plt.grid(True)
    plt.savefig('loss_curves.png')
    plt.show()
    
    desired_freqs = list(range(400, 701, 100))
    # Generate new truss units
    new_truss_units = generate_truss_units(trained_cgan, num_samples=1, desired_freqs=desired_freqs, desired_disp=5e-6)

    # Display generated truss units
    fig, axes = plt.subplots(1, 5, figsize=(20, 4))
    for i, img in enumerate(new_truss_units):
        axes[i].imshow(img[:, :, 0], cmap='gray')
        axes[i].axis('off')
    plt.tight_layout()
    plt.savefig('generated_truss_units.png')
    plt.close()

    print("Training completed and new truss units generated. Check the output images.")