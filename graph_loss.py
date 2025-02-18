import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import sys


class GraphMatrixLoss(nn.Module):
    def __init__(self, beta = 0.1):
        super(GraphMatrixLoss, self).__init__()
        self.beta = beta

    def rescale_tensor_torch(self,tensor):
        tensor_min = tensor.min()
        tensor_max = tensor.max()
        normalized_tensor = (tensor - tensor_min) / (tensor_max - tensor_min)
        return normalized_tensor

    def normalize_matrix_torch(self, matrix):
        row_sum = torch.sum(matrix, dim=1, keepdim=True)
        normalized_matrix = matrix / row_sum
        return normalized_matrix

    def remove_diagonal_flat(self,tensor,remove = False, softmax=False,keeprow = True, normrow = False):
        if tensor.dim() != 2:
            raise ValueError("Input tensor must be 2D.")
        if remove:
            n, m = tensor.shape
            mask = ~torch.eye(n, m, dtype=torch.bool)  # Create a mask where diagonal elements are False
            result = tensor[mask].flatten()
        else:
            result = tensor
        
        if remove and keeprow:
            result = result.reshape(n,m-1)
        if softmax:
            result = torch.softmax(result,dim=-1)
        if normrow:
            result = self.normalize_matrix_torch(result)
        return result
    
    def get_MST(self, dist_matrix):
        import numpy as np
        from scipy.sparse import csr_matrix
        from scipy.sparse.csgraph import minimum_spanning_tree         

        dist_matrix_numpy = dist_matrix.cpu().detach().numpy()
        # Convert to a sparse CSR matrix
        sparse_matrix = csr_matrix(dist_matrix_numpy)

        # Compute the MST
        mst_matrix_sparse = minimum_spanning_tree(sparse_matrix)

        # Convert the resulting MST back to a dense format
        mst_matrix = mst_matrix_sparse.toarray()

        # Make the matrix symmetric
        mst_matrix = np.maximum(mst_matrix, mst_matrix.T)

        # Convert back to PyTorch tensor
        mst_tensor = torch.from_numpy(mst_matrix).to(dist_matrix.device)

        return mst_tensor
    

    def forward(self, original_UD_path_matrices, original_D_path_matrices, original_dist_matrices, original_direct_matrices):
        loss = 0
        batch_size = len(original_D_path_matrices)
        for batch in range(batch_size):
            ground_UD_matrix = original_UD_path_matrices[batch]
            ground_D_matrix = original_D_path_matrices[batch]
            
            ground_positive_D_matrix = torch.where((ground_D_matrix > 0) & (ground_D_matrix < 3), ground_D_matrix, torch.tensor(0))
            ground_dist_matrix = ground_positive_D_matrix + ground_positive_D_matrix.t()
            ground_adjacent_matrix = torch.where((ground_D_matrix == 1), torch.tensor(1), torch.tensor(0))
           
            dist_matrix = original_dist_matrices[batch]
            direct_matrix = original_direct_matrices[batch]*ground_adjacent_matrix
           
            without_diag_dist_matrix = self.remove_diagonal_flat(dist_matrix,remove = True, softmax=False, keeprow = True, normrow= True)
            without_diag_ground_dist_matrix = self.remove_diagonal_flat(ground_dist_matrix,remove = True, softmax=False, keeprow = True, normrow= False)
            
            dist_loss = -torch.mean(without_diag_ground_dist_matrix*torch.log(without_diag_dist_matrix))
            direct_loss = torch.mean(nn.functional.leaky_relu(direct_matrix*(ground_adjacent_matrix.t()-ground_adjacent_matrix),negative_slope=0.01))
            loss += self.beta*dist_loss+direct_loss

        loss /= batch_size

        return loss
    
