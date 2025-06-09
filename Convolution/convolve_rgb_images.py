import numpy as np

image = np.array([
    [[1, 0, 2], [2, 1, 1], [3, 2, 0], [0, 1, 1], [1, 0, 2]],
    [[0, 1, 0], [1, 0, 1], [2, 2, 2], [3, 1, 3], [2, 0, 1]],
    [[3, 0, 2], [0, 1, 0], [1, 0, 1], [2, 2, 2], [1, 0, 0]],
    [[2, 1, 1], [1, 0, 2], [3, 3, 1], [0, 1, 0], [0, 2, 1]],
    [[1, 2, 2], [2, 1, 0], [0, 0, 1], [1, 2, 2], [2, 1, 1]]
])

image.shape

kernel=np.array([[
    [0, 1, 0], [1, -1, 1], [0, 1, 0]],
    [[1, 0, 1], [0, -1, 0], [1, 0, 1]],
    [[0, 1, 0], [1, 1, 1], [0, 1, 0]]
])

kernel.shape

def convolve_rgb(image,kernel):
  image_h,image_w,image_c=image.shape
  kernel_h,kernel_w,kernel_c=kernel.shape
  output_h=image_h-kernel_h+1
  output_w=image_w-kernel_w+1
  output=np.zeros((output_h,output_w,1))

  for k in range(0,image_c):# Apply the convolution for each channel
    for i in range(0,output_h):
      for j in range(0,output_w):
        output[i,j]=np.sum(image[i:i+kernel_h,j:j+kernel_w,k]*kernel)

  return output


output=convolve_rgb(image,kernel)
print("Convolution Output:\n", output)