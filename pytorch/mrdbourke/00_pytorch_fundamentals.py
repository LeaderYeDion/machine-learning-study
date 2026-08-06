import torch
import numpy as np

scalar = torch.tensor(1)
vector = torch.tensor([1, 2, 3])
MATRIX = torch.tensor([[1, 2, 3],
                       [4, 5, 6]])
TENSOR = torch.tensor([[[[1, 3, 5], [7, 9, 11]], [[0, 2, 4], [6, 8, 10]]]], dtype=torch.int16, device='cuda', requires_grad=False)
TENSOR_FILL_ONES = torch.ones_like(TENSOR, dtype=torch.float16, device='cuda', requires_grad=True)
TENSOR_PLUS = TENSOR + TENSOR_FILL_ONES
random_tensor = torch.rand(2, 3)

a = torch.tensor([1,2,3])
b = torch.tensor([[1], [2], [3]])
c = torch.tensor([1,2,3])

x = torch.tensor([[1,2,3]])
y = torch.tensor([[1,2,3]])
z = torch.tensor([[[[1,2,3]]]])

array = np.array([[1,2,3]])
tensor_from_np = torch.from_numpy(array)
# numpy cannot work on gpu, the code below makes error
# makes_error = TENSOR.numpy()
array_from_torch = tensor_from_np.numpy()

RANDOM_SEED = 1998
torch.manual_seed(RANDOM_SEED)
random_tensor_a = torch.rand(5, 3)

RANDOM_SEED = 1998
torch.manual_seed(RANDOM_SEED)
random_tensor_b = torch.rand(5, 3)

if __name__ == '__main__':
    print(a.ndim, b.ndim, x.ndim, z.ndim)
    # TENSORS must be on the same device in order to be calculated
    print(TENSOR.max(), TENSOR.type(torch.float16).mean(), TENSOR.argmin(), vector[vector.argmin()], b[b.argmin()], MATRIX.argmax())
    print(a + c, a*b, a.matmul(b), a.matmul(c), x.matmul(y.T))
    print(MATRIX.reshape(3, 2), z.reshape(3,), z.reshape(3,1))

    # view share the memory of original input, change the view change the original input matrix
    MATRIX_VIEW = MATRIX.view(3, 2)
    MATRIX_VIEW[0, 0] = -1
    print(MATRIX, MATRIX_VIEW)

    print(torch.stack([a, c], dim=0), torch.stack([x, y], dim=1))
    print(z.squeeze(), a.unsqueeze(1))
    print(TENSOR.shape, TENSOR.permute(3,0,2,1))
    # TENSOR[i][j] or TENSOR[i,j] are both ok, ':' symbols all dimensions.
    print(a[1], b[0,0],b[0][0], MATRIX[:, 1],MATRIX[1][:], z[0,0,0,1],z[0][0][0][1], TENSOR[0,1,1,2],  z[0,0,0,1] + TENSOR[0,1,1,2], TENSOR[0,0,1], TENSOR[:,:,:,1])

    # tensor can be converted between numpy and torch
    print(array, tensor_from_np, array_from_torch)

    # machine learning starts with a random, then adjust. To make random matrix reproducible, use random seed
    print(random_tensor_a, random_tensor_b, random_tensor_a == random_tensor_b)

    # move tensors to another device
    print(MATRIX.device, MATRIX.to("cuda"), TENSOR.to("cpu"))


