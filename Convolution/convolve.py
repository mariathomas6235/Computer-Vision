import numpy as np

image=np.array([
    [1, 2, 3, 0, 1],
    [0, 1, 2, 3, 2],
    [3, 0, 1, 2, 1],
    [2, 1, 3, 0, 0],
    [1, 2, 0, 1, 2]
])
#image.shape

kernel=np.array([
    [0, 1, 0],
    [1, -4, 1],
    [0, 1, 0]
])
#kernel.shape

def convolve(image,kernel):
  image_h,image_w=image.shape
  kernel_h,kernel_w=kernel.shape
  output_h= image_h-kernel_h+1
  output_w=image_w-kernel_w+1
  output=np.zeros((output_h,output_w)) #creates an empty array with 0s.

  for i in range(output_h):
    for j in range(output_w):
      output[i,j]=np.sum(image[i:i+kernel_h,j:j+kernel_w] * kernel)
  return output

output=convolve(image,kernel)
print("Convolution Output:\n", output)
