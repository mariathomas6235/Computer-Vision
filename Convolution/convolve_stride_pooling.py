import numpy as np

image = np.array([
    [1, 2, 3, 0, 1],
    [0, 1, 2, 3, 2],
    [3, 0, 1, 2, 1],
    [2, 1, 3, 0, 0],
    [1, 2, 0, 1, 2]
])

def convolve_with_maxpooling_and_stride(image,stride,pool_size):
  image_h,image_w=image.shape
  output_h=(image_h-pool_size)//stride+1
  output_w=(image_w-pool_size)//stride+1
  output=np.zeros((output_h,output_w))

  for i in range(0,output_h*stride,stride):
    for j in range(0,output_w*stride,stride):
      output[i//stride,j//stride]=np.max(image[i:i+pool_size,j:j+pool_size])

  return output

pooled_output=convolve_with_maxpooling_and_stride(image,stride=2,pool_size=2)
print("Max Pooling Output:\n", pooled_output)