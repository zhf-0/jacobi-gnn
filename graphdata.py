import os
import numpy as np
from scipy.sparse.linalg import spsolve
from numpy.linalg import solve, norm

import scipy.sparse as sparse
from scipy.sparse import coo_matrix

from fealpy.decorator import cartesian, barycentric
from fealpy.mesh import MeshFactory as MF
from fealpy.functionspace import LagrangeFiniteElementSpace
from fealpy.functionspace import  ParametricLagrangeFiniteElementSpace
from fealpy.boundarycondition import DirichletBC
from fealpy.tools.show import showmultirate, show_error_table

import torch
import torch_geometric.data as pygdat
from torch_geometric.utils import degree

class PDE:
    def __init__(self,x0,x1,y0,y1,blockx,blocky):
        self.x0 = x0
        self.x1 = x1
        self.y0 = y0
        self.y1 = y1
        self.xstep = (x1-x0)/blockx 
        self.ystep = (y1-y0)/blocky
        self.coef1 = 10**np.random.uniform(0.0,5.0,(blocky+1,blockx+1))
        self.coef2 = 10**np.random.uniform(0.0,5.0,(blocky+1,blockx+1))

    def domain(self):
        return np.array([self.x0, self.x1,self.y0, self.y1])
    
    @cartesian
    def solution(self, p):
        """ 
		The exact solution 
        Parameters
        ---------
        p : 
        Examples
        -------
        p = np.array([0, 1], dtype=np.float64)
        p = np.array([[0, 1], [0.5, 0.5]], dtype=np.float64)
        """
        x = p[..., 0]
        y = p[..., 1]
        pi = np.pi
        val = np.cos(pi*x)*np.cos(pi*y)
        return val # val.shape == x.shape

    @cartesian
    def source(self, p):
        """ 
		The right hand side of convection-diffusion-reaction equation
        INPUT:
            p: array object,  
        """
        x = p[..., 0]
        y = p[..., 1]
        pi = np.pi
        val = 12*pi*pi*np.cos(pi*x)*np.cos(pi*y) 
        val += 2*pi*pi*np.sin(pi*x)*np.sin(pi*y) 
        val += np.cos(pi*x)*np.cos(pi*y)*(x**2 + y**2 + 1) 
        val -= pi*np.cos(pi*x)*np.sin(pi*y) 
        val -= pi*np.cos(pi*y)*np.sin(pi*x)
        return val

    @cartesian
    def gradient(self, p):
        """ 
		The gradient of the exact solution 
        """
        x = p[..., 0]
        y = p[..., 1]
        pi = np.pi
        val = np.zeros(p.shape, dtype=np.float64)
        val[..., 0] = -pi*np.sin(pi*x)*np.cos(pi*y)
        val[..., 1] = -pi*np.cos(pi*x)*np.sin(pi*y)
        return val # val.shape == p.shape

    @cartesian
    def diffusion_coefficient(self, p):
        x = p[..., 0]
        y = p[..., 1]
        xidx = x//self.xstep
        xidx = xidx.astype(np.int)
        yidx = y//self.ystep 
        yidx = yidx.astype(np.int)

        shape = p.shape+(2,)
        val = np.zeros(shape,dtype=np.float64)
        val[...,0,0] = self.coef1[xidx,yidx]
        # val[...,0,0] = 10.0
        val[...,0,1] = 1.0
        val[...,1,0] = 1.0
        val[...,1,1] = self.coef2[xidx,yidx]
        # val[...,1,1] = 2.0
        return val

    @cartesian
    def convection_coefficient(self, p):
        return np.array([-1.0, -1.0], dtype=np.float64)

    @cartesian
    def reaction_coefficient(self, p):
        x = p[..., 0]
        y = p[..., 1]
        return 1 + x**2 + y**2

    @cartesian
    def dirichlet(self, p):
        return self.solution(p)


def GenerateMat(nx,ny,blockx,blocky):
    pde = PDE(0,1,0,1,blockx,blocky)
    domain = pde.domain()
    mesh = MF.boxmesh2d(domain, nx=nx, ny=ny, meshtype='quad',p=1)

    # space = LagrangeFiniteElementSpace(mesh, p=1)
    space = ParametricLagrangeFiniteElementSpace(mesh, p=1)
    # NDof = space.number_of_global_dofs()
    uh = space.function() 	
    A = space.stiff_matrix(c=pde.diffusion_coefficient)
    # B = space.convection_matrix(c=pde.convection_coefficient)
    # M = space.mass_matrix(c=pde.reaction_coefficient)
    F = space.source_vector(pde.source)
    # A += B 
    # A += M
    
    bc = DirichletBC(space, pde.dirichlet)
    A, F = bc.apply(A, F, uh)

    eps = 10**(-15)
    A.data[ np.abs(A.data) < eps ] = 0
    A.eliminate_zeros()

    # x = spsolve(A, F)
    # b = A.dot(x)
    # err = norm(b-F)
    # if err < 10**(-8):
    #     print('solve is ok')
    
    x = np.random.uniform(0,10,A.shape[0])
    F = A.dot(x)
    return A, x, F



class GraphData(torch.utils.data.Dataset):
    def __init__(self, num, process=0, if_float=True, transform=None):
        self.num = num
        self.if_float = if_float
            
        self.transform = transform
        self.root_path_graph = './GraphData'
            
        os.makedirs(self.root_path_graph,exist_ok=True)

        if process:
            self.Process()

    def Process(self):
        print('begin to process')
        graph_template = self.root_path_graph+'/graph{}.dat'
        for idx in range(self.num):
            graph_path = graph_template.format(idx)
            if os.path.exists(graph_path):
                continue
            
            print(f'begin to create matrix {idx}')
            A,x,b = GenerateMat(50,50,10,10)
        
            row, col = A.nonzero()

            # diag = A.diagonal()
            # A.data = A.data / diag[row]

            row = torch.from_numpy(row.astype(np.int64))
            col = torch.from_numpy(col.astype(np.int64))
            # edge_index = torch.stack((row,col),0)
            edge_index = torch.stack((col,row),0)

            edge_weight = torch.from_numpy( A.data ).unsqueeze(1)
            x0 = degree(col,A.shape[0],dtype=torch.float64).unsqueeze(1)
            y = torch.from_numpy(b).unsqueeze(1)
            sol = torch.from_numpy(x).unsqueeze(1)
            if self.if_float:
                edge_weight = edge_weight.float()
                x0 = x0.float()
                y = y.float()
                sol = sol.float()

            graph = pygdat.Data(x=x0,edge_index = edge_index,edge_weight = edge_weight,y = y)
            graph.mat_id = idx
            graph.sol = sol
            torch.save(graph,graph_path)


    def __len__(self):
        return self.num

    def __getitem__(self, idx):
        graph = torch.load(os.path.join(self.root_path_graph,'graph{}.dat'.format(idx)))
        
        if self.transform:
            graph = self.transform(graph)

        return graph


