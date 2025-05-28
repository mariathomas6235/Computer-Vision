''' Mannual conversion of RGB image to greyscale using pytorch without any available functions '''

import torch
import matplotlib.pyplot as plt
import torchvision.transforms as T
from PIL import Image

def conv(pil_image):
  transform=T.ToTensor() #converts the PIL image (shape: H x W x C) into a PyTorch tensor of shape: (C, H, W)
  image_tensor=transform(image)
  #0.2989×R+0.5870×G+0.1140×B
  gray_image_tensor = 0.2989 * image_tensor[0, :, :] + 0.5870 * image_tensor[1, :, :] + 0.1140 * image_tensor[2, :, :]
  #gray_image_tensor is 2D (just height & width), so we add a dummy channel using unsqueeze(0).Now shape becomes (1, H, W) — treated as a grayscale image.
  gray_image=T.ToPILImage()(gray_image_tensor.unsqueeze(0))

image=Image.open('/content/WhatsApp Image 2025-04-20 at 12.35.50 PM.jpeg')
gray_image=conv(image)
plt.imshow(gray_image, cmap='gray')
plt.axis('off')
plt.title('Grayscale Image (Manual Conversion)')
plt.show()
