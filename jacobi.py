import numpy as np
from numpy.linalg import  norm
import torch
from scipy.sparse import coo_matrix

def ReadMat(filename):
    graph = torch.load(filename)
    num_row = graph.y.shape[0]
    torch_col, torch_row = graph.edge_index
    row_vec = torch_row.numpy()
    col_vec = torch_col.numpy()
    val_vec = graph.edge_weight.squeeze(1).numpy()

    coo_A = coo_matrix((val_vec,(row_vec,col_vec)),shape=(num_row, num_row) )
    b = graph.y.squeeze(1).numpy()
    sol = graph.sol.squeeze(1).numpy()
    

    matfile = './coo.txt'
    matcon = [f'{num_row} {num_row} {row_vec.shape[0]} \n']
    for i in range(row_vec.shape[0]):
        line = f'{row_vec[i]} {col_vec[i]} {val_vec[i]} \n'
        matcon.append(line)
        
    with open(matfile,'w') as f:
        f.writelines(matcon)

    bfile = './b.txt'
    bcon = [f'{num_row} \n']
    for i in range(num_row):
        bcon.append(f'{b[i]} \n')

    with open(bfile,'w') as f:
        f.writelines(bcon)

    return coo_A,b,sol

def Jacobi(A,b,sol):
    diag = A.diagonal()
    x0 = np.zeros(A.shape[0])
    num_iter = 100
    tol = 10**(-8)

    for i in range(num_iter):
        Ax = A.dot(x0)
        x1 = b/diag + x0 - Ax/diag
        r = A.dot(x1)-b
        norm_r = norm(r)
        mse = np.mean(r**2)
        print(f'iter {i}: l2 residual = {norm_r}')
        print(f'iter {i}: mse residual = {mse}')
        if norm_r < tol:
            break
        x0 = x1

if __name__ == '__main__':
    A,b,sol = ReadMat('./GraphData/graph0.dat')
    Jacobi(A,b,sol)
