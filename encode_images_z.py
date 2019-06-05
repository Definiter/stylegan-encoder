import os
import pickle
import numpy as np
import PIL.Image
import dnnlib
import dnnlib.tflib as tflib
import config
import tensorflow as tf

image_name = 'black_woman_01.png'
experiment_folder = './experiments/'
image_folder = experiment_folder + 'aligned_images/'
image_path = image_folder + image_name

experiment_name = 'L2_z'
image_output_folder = experiment_folder + experiment_name + '_images/'
latent_output_folder = experiment_folder + experiment_name + '_latent/'
iteration_output_folder = image_output_folder + image_name + '_iter/'

image_size = (1024, 1024)
learning_rate = 0.001
num_iter = 1000

tflib.init_tf()
url = 'https://drive.google.com/uc?id=1MEGjdvVpUsu1jB4zrXZN7Y4kBBOzizDQ' # karras2019stylegan-ffhq-1024x1024.pkl
with dnnlib.util.open_url(url, cache_dir=config.cache_dir) as f:
    _G, _D, Gs = pickle.load(f)

# Create target image variable 
image = PIL.Image.open(image_path)
image.resize(image_size, PIL.Image.ANTIALIAS)
image = np.array(image, dtype=np.float32)
with tf.variable_scope("recovery_scope", reuse=tf.AUTO_REUSE):
    target_image_variable = tf.get_variable('target_image', 
                                            dtype=tf.float32,
                                            initializer=tf.constant(image))
print(target_image_variable)
target_variable = tf.expand_dims(target_image_variable, 0)
# target_variable = tflib.convert_images_from_uint8(target_variable, drange=[-1,1], nhwc_to_nchw=True)
print(target_variable)

# Create initial latent variable 
# z = tf.Variable(np.random.randn(1, Gs.input_shape[1]), dtype=tf.float32)
with tf.variable_scope("recovery_scope", reuse=tf.AUTO_REUSE):
    z = tf.get_variable('learnable_latents',
                        shape=(1, Gs.input_shape[1]),
                        dtype=tf.float32,
                        initializer=tf.initializers.random_normal())
print(z)

# Create output image variable
# output_variable = Gs.get_output_for(z, None, is_validation=True, randomize_noise=False)
output_variable = Gs.get_output_for(z, None, truncation_psi=0.7, randomize_noise=False)
# formatted_output_variable = tflib.convert_images_to_uint8(output_variable, drange=[-1,1], nchw_to_nhwc=True)
formatted_output_variable = tflib.convert_images_to_uint8(output_variable, 
                                                          drange=[-1,1], 
                                                          nchw_to_nhwc=True, 
                                                          uint8_cast=False)
formatted_output_variable_uint8 = tf.saturate_cast(formatted_output_variable, tf.uint8)


print(output_variable)
print(formatted_output_variable)

# Loss
# loss = tf.losses.mean_squared_error(labels=target_variable, predictions=output_variable)
loss = tf.losses.mean_squared_error(labels=target_variable, predictions=formatted_output_variable)
print(loss)

# Optimizer
optimizer = tf.train.GradientDescentOptimizer(learning_rate)
# optimizer = tf.train.AdamOptimizer(learning_rate)
train_op = optimizer.minimize(loss, var_list=z)
print(optimizer)
print(train_op)

# Session
sess = tf.get_default_session()
# sess.run(tf.global_variables_initializer())
sess.run(z.initializer)
sess.run(target_image_variable.initializer)
formatted_output_value_uint8 = None
z_value = None
if not os.path.exists(iteration_output_folder):
    os.mkdir(iteration_output_folder)
for it in range(num_iter):
    _, loss_value, z_value, formatted_output_value_uint8, formatted_output_value, target_value = sess.run((train_op, loss, z, formatted_output_variable_uint8, formatted_output_variable, target_variable))
    print("-------")    
    print(it)
    PIL.Image.fromarray(formatted_output_value_uint8[0], 'RGB').save((iteration_output_folder + '{:05d}.png').format(it))
    print("Loss: ", loss_value)
#     print("formatted_output_value: ", formatted_output_value)
#     print("target_value: ", target_value)

# Save image
PIL.Image.fromarray(formatted_output_value_uint8[0], 'RGB').save(image_output_folder + image_name)
np.save(os.path.join(latent_output_folder + image_name + '.npy'), z_value)

