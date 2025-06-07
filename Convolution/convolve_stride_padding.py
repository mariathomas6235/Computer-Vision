import numpy as np

kernel = np.array([
    [0, 1, 0],
    [1, -4, 1],
    [0, 1, 0]
])
image=np.array([
    [1, 2, 3, 0, 1],
    [0, 1, 2, 3, 2],
    [3, 0, 1, 2, 1],
    [2, 1, 3, 0, 0],
    [1, 2, 0, 1, 2]
])

def pad_image(image,pad):
    return np.pad(image,pad,mode='constant',constant_values=0)


def convolve_with_padding_and_stride(image,kernel,padding=0,stride=1):
    if padding>0:
        image=pad_image(image,padding)

    image_h,image_w=image.shape
    kernel_h,kernel_w=kernel.shape
    output_h=(image_h-kernel_h)//stride+1
    output_w=(image_w-kernel_w)//stride+1
    output=np.zeros((output_h,output_w))

    for i in range(0,output_h*stride,stride):
        for j in range(0,output_w*stride,stride):
            output[i//stride,j//stride]=np.sum(
                image[i:i+kernel_h,j:j+kernel_w]*kernel
            )

    return output


padded_output=convolve_with_padding_and_stride(image,kernel,stride=1,padding=1)
print("Padded Convolution Output:\n", padded_output)