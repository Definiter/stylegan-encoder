import os
import pickle
import numpy as np
import PIL.Image
import dnnlib
import dnnlib.tflib as tflib
import config


def timing_linear(t):
    return t

def timing_ease_in_out(t):
    return (t ** 2) / (t ** 2 + (1 - t) ** 2)

def generate_sequence(Gs, num_key_frame, num_interval, random_seed=0, sub_dir = '', timing_function = timing_linear):
    
    
#     rnd = np.random.RandomState(random_seed)
    
#     # Latents of key frames.
#     key_latents = []
#     for _ in range(num_key_frame):
#         key_latents.append(rnd.randn(1, Gs.input_shape[1]))
    
    # Latents of key frames.
    seeds = [153, 76, 113, 29, 28]
    key_latents = []
    for i in range(len(seeds)):
        rnd = np.random.RandomState(seeds[i])
        key_latents.append(rnd.randn(1, Gs.input_shape[1]))
    
    # Interpolated latents.
    latents = []
    if len(key_latents) > 1:
        for i in range(len(key_latents) - 1):
            start = key_latents[i]
            end = key_latents[i + 1]
            latents.append(start)
            for j in range(num_interval):
                percentage = float(j + 1) / (num_interval + 1)
                interpolated_latent = start + (end - start) * timing_function(percentage)
                latents.append(interpolated_latent)
    latents.append(key_latents[-1])
    
    for i in range(len(latents)):
        print(i)
        # Generate image.
        fmt = dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True)
        images = Gs.run(latents[i], None, truncation_psi=0.7, randomize_noise=False, output_transform=fmt)

        # Save image.
        path = config.result_dir + sub_dir
        os.makedirs(path, exist_ok=True)
        png_filename = os.path.join(path, '{}.png'.format(i))
        PIL.Image.fromarray(images[0], 'RGB').save(png_filename)
    

tflib.init_tf()
url = 'https://drive.google.com/uc?id=1MEGjdvVpUsu1jB4zrXZN7Y4kBBOzizDQ' # karras2019stylegan-ffhq-1024x1024.pkl
# url = 'https://drive.google.com/uc?id=1MOSKeGF0FJcivpBI7s63V9YHloUTORiF' # stylegan-bedrooms-256x256.pkl
with dnnlib.util.open_url(url, cache_dir=config.cache_dir) as f:
    _G, _D, Gs = pickle.load(f)
    

generate_sequence(Gs, 5, 60, 7, '/ffhq_final', timing_linear)

# for seed in range(40, 50):
#     generate_sequence(Gs, 5, 60, seed, '/ffhq_seed/ffhq_' + str(seed), timing_linear)
    
    