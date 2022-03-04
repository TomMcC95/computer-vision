import matplotlib.pyplot as plt
from src.Basler import BaslerCamera

# Connect to camera
camera = BaslerCamera()

# Aquire single image
img = camera.get_single_image()
plt.imshow(img)
plt.show()
print(f'Size of image: {img.shape}')

# Acquire single strem of images
image_array = camera.get_stream(50)
print(f'Size of stream of images: {image_array.shape}')
plt.imshow(image_array[:, :, :, 0])
plt.show()

