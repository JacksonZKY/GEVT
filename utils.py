import torch
import random
import torch.nn.functional as F
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree

                
def matrix_overlap(ground_matrices, result_matrices):
    batch_size = len(ground_matrices)
    all_EO = 0
    all_num = 0
    all_undirected_EO = 0
    all_VEO = 0
    for batch in range(batch_size):
        ground_edge_num = 0
        result_edge_num = 0
        good_edge_num = 0
        good_undirected_edge_num = 0
        ground_matrix = ground_matrices[batch]
        result_matrix = result_matrices[batch]

        shape0,shape1 = ground_matrix.shape[0],ground_matrix.shape[1]
        
        for i in range(shape0):
            for j in range(shape1):
                if i == j:
                    continue
                ground = ground_matrix[i][j]
                output = -1
                if result_matrix[i][j] > 0:
                    output = 1
                    result_edge_num += 1
                if ground == 1:
                    ground_edge_num+=1
                    if output > 0:
                        good_edge_num += 1
                    if result_matrix[i][j] != 0:
                        good_undirected_edge_num += 1

        undirected_EO = 2*good_undirected_edge_num/(ground_edge_num+result_edge_num)
        EO = 2*good_edge_num/(ground_edge_num+result_edge_num)

        node_num = shape0
        VEO = 2*(node_num+ good_edge_num)/(2*node_num+ground_edge_num+result_edge_num)

        all_EO += EO
        all_undirected_EO += undirected_EO
        all_VEO += VEO
        all_num += 1

    return all_undirected_EO, all_EO, all_VEO, all_num



def get_MST(dist_matrix):
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


def get_result_matrices(self,original_dist_matrices, original_direct_matrices):
    batch_size = len(original_dist_matrices)
    result_matrices = []
    for batch in range(batch_size):
        dist_matrix = original_dist_matrices[batch]
        direct_matrix = original_direct_matrices[batch]
        MST_matrix = get_MST(dist_matrix)
        result_matrix = MST_matrix*direct_matrix
        result_matrices.append(result_matrix)
    return result_matrices

    
