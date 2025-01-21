import gmsh
import numpy as np
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from scipy import sparse
import time
import json
from random import randint, randrange
from scipy.sparse import linalg as sla
from scipy.sparse import coo_matrix

import os

import cvxpy as cp

# from NN_TopOpt.mesh_utils import LoadedMesh2D
from mesh_utils import LoadedMesh2D

def BuildIkFunc0():
  return lambda me,k: np.array([2*me[k,0],2*me[k,0]+1,
                                2*me[k,1],2*me[k,1]+1,
                                2*me[k,2],2*me[k,2]+1])

def Hooke2DP1(la, mu):
    C=np.array([[la+2*mu, la , 0],
                [la, la+2*mu, 0 ],
                [0, 0, mu]])
    return C

def ElemStiffElasMatBa2DP1(ql,V,C):
    #   https://www.math.univ-paris13.fr/~cuvelier/docs/Recherch/FEM2Dtoolbox/doc/V1.2b3/_elem_stiff_elas_mat_p1_ba_8m_source.html
  """ Returns the element elastic stiffness matrix :math:`\\mathbb{K}^e(T)`
  for a given tetrahedron :math:`T`  in the local *alternate* basis :math:`\\mathcal{B}_a`

  :param ql: contains the four vertices of the tetrahedron,
  :type ql: :math:`4 \\times 2` *numpy* array
  :param V: volume of the tetrahedron
  :type V: float
  :param H: Elasticity tensor, :math:`\\mathbb{H}`.
  :type H: :math:`6 \\times 6` *numpy* array
  :returns: :math:`\\mathbb{K}^e(T)` in :math:`\\mathcal{B}_a` basis.
  :type: :math:`12 \\times 12` *numpy* array of floats.
  """
  u=ql[1]-ql[2]
  v=ql[2]-ql[0] 
  w=ql[0]-ql[1]
  
  B= np.array([[u[1],  0 ,   v[1],  0,    w[1],  0],
               [0,    -u[0], 0,    -v[0], 0,    -w[0]],
               [-u[0], u[1], -v[0], v[1], -w[0], w[1]]])
  
  return np.dot(np.dot(B.T,C),B)/(4*V)

GetI2DP1 = BuildIkFunc0()

class TopOptimizer2D:
    def __init__(self, method_dict, args, activate_method = True) -> None:
        
        self.problem_name = args['problem_name']
        try:
            self.image_size = args['image_size']
        except:
            self.image_size = 12
        # load problems config
        with open('../test_problems/problems.json', 'r') as fp:
            problem_list = json.load(fp)

        self.problem_args = problem_list[self.problem_name]

        self.Th = LoadedMesh2D(f'../{self.problem_args["meshfile"]}')

        # to save indeces and K_e for assembling global stiffness matrix
        self.ik = []
        self.jk = []
        self.K_sep = []
        self.edof = []

        # to save boundary conditions and load
        self.fixed_dof = []
        self.loaded_dof = []
        self.moved_dof = []

        self.nme = self.Th.me.shape[0]   # number of elements
        self.ndof = 2*self.Th.q.shape[0] # number of degrees of freedoms

        self.u = np.zeros((self.ndof,))  # vector of displacments
        self.f = np.zeros((self.ndof,))  # vector of loads
        self.ce = np.ones(self.Th.me.shape[0]) # vector of compliances

        # get Hooke matrix
        la = 1.5
        mu = 0.5
        self.C=Hooke2DP1(la,mu)

        self.build_stiffness_matrix()
        self.build_constraints()
        self.apply_loads()

        # update initial conditions
        self.dofs = np.arange(self.ndof)

        self.fixed_dof = np.unique(np.array(self.fixed_dof))
        self.moved_dof = np.array([])
        self.moved_fixed_dof = np.concatenate((self.moved_dof, self.fixed_dof)).astype(int)
        self.free_dof = np.setdiff1d(self.dofs, self.fixed_dof)
        self.free_dof = np.setdiff1d(self.free_dof, self.moved_dof)

        self.jk = np.array(self.jk)
        self.ik = np.array(self.ik)
        self.iK = self.ik.flatten()
        self.jK = self.jk.flatten()
        self.edof = np.array(self.edof, dtype=int)
        self.K_sep = np.array(self.K_sep)

        #for SIMP type methods
        self.penal = args["penal"]
        self.Emin = 0.0001
        self.Emax = 1
        self.obj = 0

        if activate_method:
            self.meth_args = {"Th": self.Th,
                            "penal": self.penal,
                            "args": args,
                            "problem_config": self.problem_args,
                            "Emin": self.Emin,
                            "Emax": self.Emax}

            self.method = method_dict[args['method']](self.meth_args)

        ### initialize logger file
        self.log_file_name = 'experiments/'+''.join(["{}".format(randint(0, 9)) for num in range(0, 10)])+'.json'
        meta = {
            'args': args,
            'problem': self.problem_args,
            'data': time.strftime("%Y-%m-%d"),
            'time': time.strftime("%H:%M:%S", time.localtime()),
            'iter_meta': {}
        }

        if not os.path.exists('experiments'):
            os.makedirs('experiments')
        
        with open(self.log_file_name, 'w') as fp:
            json.dump(meta, fp)

        # print(self.log_file_name)
        self.F_free_to_invest = None
        self.K_free_to_invest = None

    def build_stiffness_matrix(self):
        # create the elasticity problem
        for k in tqdm(range(self.Th.me.shape[0])): # iterate each element
            # get stiffness matrix for element
            E=ElemStiffElasMatBa2DP1(self.Th.q[self.Th.me[k]],self.Th.areas[k], self.C)

            # get indeces of degrees of freedoms for the element
            I=GetI2DP1(self.Th.me,k)

            # to save indeces and K_e separately  
            ik_e = []
            jk_e = []
            K_e = []
            edof_e = []

            # iterate elements in K_e to save them and their global indeces
            # of dofs to assembly global stiffness matrix
            for il in range(0,6):
                i=I[il]
                edof_e.append(i)

                for jl in range(0,6):
                    j=I[jl]
                        
                    ik_e.append(i)
                    jk_e.append(j)
                    K_e.append(E[il,jl])

            # save elements of global stiffness matrix in separted form
            self.ik.append(ik_e)
            self.jk.append(jk_e)
            self.K_sep.append(K_e)
            self.edof.append(edof_e)

    def build_constraints(self):
        # split coords
        node_range = np.arange(self.Th.q.shape[0])
        
        for fixed_case in ["fixed_x", "fixed_y", "fixed_xy"]:
            
            case_list = self.problem_args[fixed_case]
            for case in case_list:
                
                # x component
                if isinstance(case[0], list):
                    x_bids = np.logical_and(self.Th.q[:, 0] >= case[0][0], self.Th.q[:, 0] <= case[0][1])
                else:
                    x_bids = np.isclose(self.Th.q[:, 0], case[0])

                # y component
                if isinstance(case[1], list):
                    y_bids = np.logical_and(self.Th.q[:, 1] >= case[1][0], self.Th.q[:, 1] <= case[1][1])
                else:
                    y_bids = np.isclose(self.Th.q[:, 1], case[1])

                bc_bids = np.logical_and(x_bids, y_bids)
                print("Fixed case: ", fixed_case, case, bc_bids.sum(), y_bids.sum())
                node_ids = node_range[bc_bids]

                for node_id in node_ids:
                    if fixed_case == "fixed_x":
                        # print("Fixed_node_x: ", node_id)
                        self.fixed_dof.append(2*node_id)
                    elif fixed_case == "fixed_y":
                        # print("Fixed_node_y: ", node_id)
                        self.fixed_dof.append(2*node_id+1)
                    else:
                        # print("Fixed_node: ", node_id)
                        self.fixed_dof.append(2*node_id)
                        self.fixed_dof.append(2*node_id+1)

    def apply_loads(self):
        node_range = np.arange(self.Th.q.shape[0])
        load_list = self.problem_args["loads"]
        for case in load_list:
            # x component
            if isinstance(case[0][0], list):
                x_bids = np.logical_and(self.Th.q[:, 0] >= case[0][0][0], self.Th.q[:, 0] <= case[0][0][1])
            else:
                x_bids = np.isclose(self.Th.q[:, 0], case[0][0])

            # y component
            if isinstance(case[0][1], list):
                y_bids = np.logical_and(self.Th.q[:, 1] >= case[0][1][0], self.Th.q[:, 1] <= case[0][1][1])
            else:
                y_bids = np.isclose(self.Th.q[:, 1], case[0][1])

            bc_bids = np.logical_and(x_bids, y_bids)
            node_ids = node_range[bc_bids]

            # apply loads distributed over nodes
            self.f[node_ids*2] = case[1][0]/node_ids.shape[0]
            self.f[node_ids*2+1] = case[1][1]/node_ids.shape[0]

        print("Loaded loads: ", self.f[self.f != 0].shape)

    def update_meth_args(self):
        self.meth_args["ce"] = self.ce

    def log_meta(self):
        iteration = self.method.global_i
        iteration_meta = self.method.meta

        # with open(self.log_file_name, 'r') as fp:
        #     meta = json.load(fp)

        # meta["iter_meta"][iteration] = iteration_meta

        # with open(self.log_file_name, 'w') as fp:
        #     json.dump(meta, fp)

    def save_data(self, directory):
        np.save(f"{directory}/K_sep.npy", self.K_sep)
        np.save(f"{directory}/free_dof.npy", self.free_dof)
        np.save(f"{directory}/moved_fixed_dof.npy", self.moved_fixed_dof)
        np.save(f"{directory}/f.npy", self.f)
        np.save(f"{directory}/ik.npy", self.ik)
        np.save(f"{directory}/jk.npy", self.jk)

    def plot_final_result(self, geometry_features = None, filename=None):
        self.Th.plot_topology(self.x, image_size=self.image_size, geometry_features=geometry_features, filename=filename)

    def save_solution(self, directory):
        np.save(f"{directory}/u.npy", self.u)
        np.save(f"{directory}/x.npy", self.x)
    
    def optimize(self):
        self.log_meta()
        counter = 0
        while not self.method.stop_flag:
            counter += 1
            xPhys = self.method.get_x(self.meth_args)

            # build global stiffness matrix
            sK=(self.K_sep.T*(self.Emin+(xPhys)**self.penal*(self.Emax-self.Emin))).flatten(order='F')
            # print(sK.shape)
            # print(self.iK.shape)
            # print(self.ndof)
            K = coo_matrix((sK,(self.iK,self.jK)),shape=(self.ndof,self.ndof)).tocsc()

            # compute RHS
            # print("compute RHS")
            # F_free = self.f[self.free_dof] - (K[self.free_dof,:][:,np.concatenate((self.moved_dof, self.fixed_dof)).astype(int)] @ self.u[np.concatenate((self.moved_dof, self.fixed_dof)).astype(int)])
            F_free = self.f[self.free_dof] - (K[self.free_dof,:][:,self.moved_fixed_dof] @ self.u[self.moved_fixed_dof])
            
            K_free = K[self.free_dof,:][:,self.free_dof]

            # if counter == 30:
            #     self.F_free_to_invest = F_free.copy()
            #     self.K_free_to_invest = K_free.copy()
            #     break

            # compute SLE
            # print("compute SLE")
            K_free = K_free.tocsc() #  Need CSR for SuperLU factorisation
            lu = sla.splu(K_free)
            self.u[self.free_dof] = lu.solve(F_free)

            # show displacement
            # self.Th.plot_displacement(self.u)
            # break
            # self.Th.plot_topology(xPhys, image_size=self.image_size)

            # print("U max", self.u.max())
            # compute compliance vector
            # print("Compute compliance vecotor ...")
            for i, (edof_e, K_e_flatten) in enumerate(zip(self.edof, self.K_sep)): # iterate each element
                u_e = self.u[edof_e] # vector of displacements for the element
                K_e = K_e_flatten.reshape(6, 6) # upload K_e
                self.ce[i] = np.dot(np.dot(u_e.T, K_e), u_e) # compute compliance for the element

            
            self.obj=((self.Emin+xPhys**self.penal*(self.Emax-self.Emin))*self.ce).sum() # global compliance
            # print("Computer obj ...: ", self.obj)

            self.log_meta()
            self.update_meth_args()

        self.x = xPhys
                   
class TopOptimizer2D_ADMM(TopOptimizer2D):
    def __init__(self, method_dict, args, activate_method = True) -> None:
        super().__init__(method_dict, args, activate_method)

        self.gamma_1 = args["gamma_1"]
        self.gamma_2 = args["gamma_2"]
        self.gamma_3 = args["gamma_3"]

        self.w = np.zeros((self.ndof,))  # vector of displacments
        self.tilde_mu = np.zeros((self.ndof,))  # vector of displacments

        self.gamma_3_I = self.gamma_3 * sparse.eye(self.ndof)

        # self.K_sep_torch = torch.tensor(self.K_sep)
        # self.indeces_K_torch = torch.tensor([self.iK, self.jK])
        self.indeces_K = np.array([self.iK, self.jK])
        
        self.meth_args = {"Th": self.Th,
                          "penal": self.penal,
                          "args": args,
                          "problem_config": self.problem_args,
                          "Emin": self.Emin,
                          "Emax": self.Emax,
                          "ndof": self.ndof,
                          "K_sep": self.K_sep,
                          "indeces_K": self.indeces_K,
                          "f": self.f}

        self.method = method_dict[args['method']](self.meth_args)

    def update_meth_args(self):
        self.meth_args["ce"] = self.ce
        self.meth_args["u"] = self.u

    def optimize(self):
        self.log_meta()
        xPhys = self.method.x_init

        while not self.method.stop_flag:
            # update u
            sK=(self.K_sep.T*(self.Emin+(xPhys)**self.penal*(self.Emax-self.Emin))).flatten(order='F')
            K = coo_matrix((sK,(self.iK,self.jK)),shape=(self.ndof,self.ndof)).tocsc()

            A = self.gamma_1*(K.T @ K) + self.gamma_3_I
            # print("A: ", A.shape)
            # print("K: ", K.shape)
            # print("self.w: ", self.w.shape)
            # print("self.tilde_mu: ", self.tilde_mu.shape)
            RHS = self.gamma_2*(K.T @ self.f) + self.gamma_3*(self.w - self.tilde_mu)

            RHS_free = RHS[self.free_dof] - (A[self.free_dof,:][:,self.moved_fixed_dof] @ self.u[self.moved_fixed_dof])            
            A_free = A[self.free_dof,:][:,self.free_dof]

            A_free = A_free.tocsc() #  Need CSR for SuperLU factorisation
            lu = sla.splu(A_free)
            self.u[self.free_dof] = lu.solve(RHS_free)

            # update xPhys
            self.update_meth_args()
            xPhys = self.method.get_x(self.meth_args).detach().numpy()

            # update w
            self.w = -self.f/self.gamma_3 + self.u + self.tilde_mu

            # update tilde_mu
            self.tilde_mu = self.tilde_mu + self.gamma_3*(self.u - self.w)

            self.Th.plot_topology(xPhys)
            self.log_meta()
            
class TopOptimizer2D_LP(TopOptimizer2D):
    def __init__(self, method_dict, args, activate_method = True) -> None:
        super().__init__(method_dict, args, activate_method)

        self.volumes = self.Th.areas
        self.assemble_t()

    def assemble_t(self):
        t_list = []
        for k in tqdm(range(self.nme), desc="Assembling t"):
            sk=self.K_sep[k].T.flatten(order='F')
            # print(sK.shape)
            # print(self.iK.shape)
            # print(self.ndof)
            K = coo_matrix((sk,(self.ik[k],self.jk[k])),shape=(self.ndof,self.ndof)).tocsc()

            # compute RHS
            # print("compute RHS")
            # F_free = self.f[self.free_dof] - (K[self.free_dof,:][:,np.concatenate((self.moved_dof, self.fixed_dof)).astype(int)] @ self.u[np.concatenate((self.moved_dof, self.fixed_dof)).astype(int)])
            F_free = self.f[self.free_dof] - (K[self.free_dof,:][:,self.moved_fixed_dof] @ self.u[self.moved_fixed_dof])
            K_free = K[self.free_dof,:][:,self.free_dof]

            t = F_free.T @ K_free.T @ F_free

            t_list.append(t)

        self.t = np.array(t_list)

        
def oc(nme,x, v, vol_goal,dc,dv,g):
    """
    The function finds next X for optimization in SIMP method
    and upadates optimality criterion.
    """
    l1=0
    l2=1e9
    move=0.2
    # reshape to perform vector operations
    xnew=np.zeros(nme)
    while (l2-l1)/(l1+l2)>1e-3:
        lmid=0.5*(l2+l1)
        # print(lmid)
        xnew= np.maximum(0.001,np.maximum(x-move,np.minimum(1.0,np.minimum(x+move,x*np.sqrt(-dc/dv/lmid)))))

        # gt=g+np.sum((dv*(xnew-x)))
        gt = np.sum(xnew * v) - vol_goal

        if gt>0 :
            l1=lmid
        else:
            l2=lmid
    return xnew, gt

def filter_matrix(points, rmin):

  x_array = points[:, 0]
  y_array = points[:, 1]

  indeces = np.arange(points.shape[0])
    
  # H = sparse.csr_matrix((points.shape[0], points.shape[0]))
  H = sparse.lil_matrix((points.shape[0], points.shape[0]))
  print("Build filter matrix ...")
  for i, point in enumerate(tqdm(points)):
      x_c = point[0]
      y_c = point[1]

      x_c_r = (x_array <= x_c + rmin)
      x_c_l = (x_array >= x_c - rmin)

      y_c_r = (y_array <= y_c + rmin)
      y_c_l = (y_array >= y_c - rmin)

      result_mask = x_c_r & x_c_l & y_c_r & y_c_l

      nearest_indeces = indeces[result_mask]
      distances = np.sqrt((x_c - x_array[nearest_indeces])**2 + (y_c - y_array[nearest_indeces])**2)
      fac = rmin - distances
      sH_i = np.maximum(0.0,fac)

      H[i, nearest_indeces] = sH_i

  return H.tocsc()


def fit_ellipsoid(x, target_area):
    # Create variables: positive semi-definite matrix A and vector b
    A = cp.Variable((2,2), PSD=True)
    b = cp.Variable((2,))

    # Form and solve the problem
    prob = cp.Problem(cp.Minimize(-cp.log_det(A)), 
                      [cp.norm(A @ x[i] + b) <= 1 for i in range(x.shape[0])])

    prob.solve()

    # Extract the optimal value of A
    A_opt = A.value

    # Calculate area of fitted ellipsoid
    current_area = np.pi / np.sqrt(np.linalg.det(A_opt @ A_opt.T))

    # Calculate scaling factor
    scale_factor = np.sqrt(target_area / current_area)

    # Scale A matrix to achieve target area
    A_opt = A_opt / scale_factor

    Q = A_opt @ A_opt.T

    # Extract eigenvalues and eigenvectors
    eigenvals, eigenvecs = np.linalg.eigh(Q)

    # Sort eigenvalues and eigenvectors in descending order
    idx = eigenvals.argsort()[::-1]
    eigenvals = eigenvals[idx]
    eigenvecs = eigenvecs[:,idx]

    # Rotation matrix (eigenvectors)
    R = eigenvecs

    # Matrix of sigmas (square root of eigenvalues)
    Sigmas = 1/np.sqrt(eigenvals)

    # Center of the ellipsoid
    center = np.linalg.inv(A_opt) @ b.value

    return center, R, Sigmas

class SIMP_basic:
    """
    Basic SIMP method for topology optimization
    """
    def __init__(self, args) -> None:
        self.Emin = args["Emin"]
        self.Emax = args["Emax"]
        self.penal = args["penal"]
        self.volfrac = args["args"]["volfrac"]
        self.rmin = args["args"]["rmin"]
        self.Th = args["Th"]
        self.nme = self.Th.me.shape[0]
        self.x = self.volfrac * np.ones(self.Th.me.shape[0],dtype=float)
        self.x_old = self.x.copy()

        self.volumes = self.Th.areas
        self.volumes_sum = self.volumes.sum()
        self.vol_goal = self.volumes_sum*self.volfrac

        self.dv = self.Th.areas/self.Th.areas.max()
        print("check dv", (self.dv == 0).sum(), self.dv.mean())
        self.dc = np.ones(self.Th.me.shape[0])
        self.ce = np.ones(self.Th.me.shape[0])
        self.g = 0

        self.H = filter_matrix(self.Th.centroids, self.rmin)
        self.Hs = self.H.sum(1)

        self.stop_flag = False
        self.obj = 0

        self.global_i = 0
        self.meta = {'x': self.x.copy().tolist(), 'dc': self.dc.copy().tolist(), 'stop_flag': self.stop_flag}

    def get_x(self, args):

        if self.global_i == 0:
            self.global_i += 1
            return self.x.copy()
        
        ce = args["ce"]
        obj=((self.Emin+self.x**self.penal*(self.Emax-self.Emin))*ce).sum()

        self.dc[:]=(-self.penal*self.x**(self.penal-1)*(self.Emax-self.Emin))*ce
        self.dc[:] = np.asarray((self.H*(self.x*self.dc))[np.newaxis].T/self.Hs)[:,0] / np.maximum(0.001, self.x)

        self.xold = self.x.copy()
        self.x, self.g=oc(self.nme, self.x, self.volumes, self.vol_goal, self.dc, self.dv, self.g)
        # print(self.x)

        print("current volume: ", (self.x.T @ self.volumes)/self.volumes_sum)
        
        change=np.linalg.norm(self.x.reshape(self.nme,1)-self.xold.reshape(self.nme,1),np.inf)

        self.global_i += 1
        if self.global_i == 40:
            self.stop_flag = True

        self.meta = {'x': self.x.copy().tolist(),
                     'ce': ce.copy().tolist(),
                     'obj': obj,
                     'dc': self.dc.copy().tolist(),
                     'change': change,
                     'stop_flag': self.stop_flag}
        
        print(self.global_i, obj, change)

        return self.x.copy()
    
class SIMP_ADMM(torch.nn.Module):
    """
    SIMP method for topology optimization with ADMM
    """
    def __init__(self, args) -> None:
        super().__init__()

        self.Emin = args["Emin"]
        self.Emax = args["Emax"]
        self.penal = args["penal"]
        self.volfrac = args["args"]["volfrac"]
        self.rmin = args["args"]["rmin"]
        self.Th = args["Th"]
        self.gamma_2 = args["args"]["gamma_2"]
        self.ndof = args["ndof"]
        self.nme = self.Th.me.shape[0]
        self.x_init = self.volfrac * np.ones(self.Th.me.shape[0],dtype=float)
        self.x_values = torch.logit(torch.tensor(self.volfrac * np.ones(self.Th.me.shape[0],dtype=float)))
        self.W_x = torch.nn.Parameter(self.x_values)

        self.volumes = torch.tensor(self.Th.areas)
        self.volumes_sum = self.volumes.sum()
        self.vol_goal = self.volumes_sum*self.volfrac

        self.dv = self.Th.areas/self.Th.areas.max()
        print("check dv", (self.dv == 0).sum(), self.dv.mean())
        self.dc = np.ones(self.Th.me.shape[0])
        self.ce = np.ones(self.Th.me.shape[0])
        self.g = 0

        self.H = filter_matrix(self.Th.centroids, self.rmin)
        self.Hs = self.H.sum(1)

        self.stop_flag = False
        self.obj = 0

        self.K_sep = torch.tensor( args["K_sep"])
        self.indeces_K = torch.tensor(args["indeces_K"])
        self.f = torch.tensor(args["f"])

        self.global_i = 0
        self.meta = {'x': self.x_values.detach().numpy().tolist(), 'stop_flag': self.stop_flag}

        self.optimizer = torch.optim.Adam([self.W_x], lr=0.01)

    def get_x(self, args):
        u = torch.tensor(args["u"])

        for i in range(10):
            self.optimizer.zero_grad()

            rho = torch.sigmoid(self.W_x)
            # vofrac_loss = torch.nn.functional.relu(self.vol_goal - rho.T @ self.volumes)
            vofrac_loss = torch.nn.functional.relu(rho.mean() - self.volfrac)

            sK=(self.K_sep.T*(self.Emin+(rho)**self.penal*(self.Emax-self.Emin))).T.flatten()
            # print(sK.shape)
            # print(self.iK.shape)
            # print(self.ndof)
            # print(self.indeces_K.shape)
            # print(sK.shape)
            K = torch.sparse_coo_tensor(self.indeces_K, sK, size=(self.ndof, self.ndof))
            confidence_loss = torch.norm(K @ u - self.f)

            print(f'vol_loss: {vofrac_loss}, conf_loss: {confidence_loss}, rho: {rho.mean()}, {rho.min()}, {rho.max()}')

            loss = vofrac_loss + self.gamma_2*confidence_loss
            loss.backward()
            self.optimizer.step()

        self.global_i += 1

        if self.global_i == 10:
            self.stop_flag = True

        return rho

class GaussianSplattingCompliance(torch.nn.Module):
    def __init__(self, dist_means_, coords, Emin, Emax, penal, num_samples, args):
        super().__init__()

        self.sigma_min = 0.002
        self.sigma_max = 0.1
        self.det_min = 3e-06
        self.det_max = 5.5e-06
        dist_means = torch.zeros((num_samples, 2))
        init_num_samples = dist_means_.shape[0]

        x_min = coords[:, 0].min()
        x_max = coords[:, 0].max()
        y_min = coords[:, 1].min()
        y_max = coords[:, 1].max()

        self.coord_max = torch.tensor([x_max, y_max])
        self.coord_min = torch.tensor([x_min, y_min])

        self.compliance_w = args["compliance_w"]
        self.volfrac_w = args["volfrac_w"]
        self.gaussian_overlap_w = args["gaussian_overlap_w"]
        self.smooth_k = args["smooth_k"]

        # print("dist_means: ", dist_means.shape)
        # print("dist_means_: ", dist_means_.shape)
        # print("init_num_samples: ", init_num_samples)

        dist_means[:init_num_samples] = dist_means_


        # scale_x = 1/coords[:, 0].max()
        self.width_ratio = coords[:, 1].max()/coords[:, 0].max()
        self.scaler = torch.tensor([1, self.width_ratio])
        self.scale_sigma = torch.tensor(0.06)/4
        self.scale_max = self.scale_sigma*2
        self.scale_min = self.scale_sigma*0.75

        self.rotation_min = -torch.pi/2
        self.rotation_max = torch.pi/2

        print("Scale_sigma_init: ", self.scale_sigma)

        self.sigmas_ratio_max = 2.5
        self.sigmas_ratio_min = 0.5

        self.coords = coords

        # coords[:, 0] = coords[:, 0]*scale_x
        # coords[:, 1] = coords[:, 1]*scale_x

        # center_coords_normalized = torch.tensor([0.5, 0.5*self.width_ratio]).type(torch.DoubleTensor) #.float()
        # coords = (center_coords_normalized - coords) * 2.0

        # dist_means[:, 0] = dist_means[:, 0]* scale_x
        # dist_means[:, 1] = dist_means[:, 1]* scale_x

        # dist_means = (center_coords_normalized - dist_means)*2.0

        # self.coords = torch.tanh(torch.atanh(coords)).type(torch.DoubleTensor)

        # offsets_values = torch.atan(dist_means.type(torch.DoubleTensor))
        offsets_values = torch.logit((dist_means - self.coord_min)/(self.coord_max - self.coord_min)).type(torch.DoubleTensor)
        # scale_values_real = torch.ones((num_samples, 1))*self.scale_sigma
        scale_values = torch.logit((torch.ones((num_samples, 1))*self.scale_sigma - self.scale_min)/(self.scale_max - self.scale_min)).type(torch.DoubleTensor)
        sigmas_ratio_values = torch.logit((torch.ones((num_samples, 1)) - self.sigmas_ratio_min)/(self.sigmas_ratio_max - self.sigmas_ratio_min)).type(torch.DoubleTensor)
          # Initially set to 0, which corresponds to a ratio of 1 after applying tanh
        rotation_values = torch.logit((torch.zeros((num_samples, 1)) - self.rotation_min)/(self.rotation_max - self.rotation_min)).type(torch.DoubleTensor)
        
        # self.W = torch.nn.Parameter(W_values)
        # self.offset_values = torch.nn.Parameter(offsets_values)
        # W_values = torch.cat([scale_values, rotation_values, offsets_values], dim=1)
        
        self.W_scale = torch.nn.Parameter(scale_values)
        self.W_sigmas_ratio = torch.nn.Parameter(sigmas_ratio_values)
        self.W_rotation = torch.nn.Parameter(rotation_values)
        self.W_offsets = torch.nn.Parameter(offsets_values)

        left_over_size = num_samples - init_num_samples
        self.current_marker = init_num_samples

        self.persistent_mask = torch.cat([torch.ones(init_num_samples, dtype=bool),torch.zeros(left_over_size, dtype=bool)], dim=0)
        
        # self.offsets_values = torch.nn.Parameter(torch.atan(dist_means.type(torch.DoubleTensor)))
        # self.scale_values = torch.nn.Parameter(torch.logit(torch.ones((num_samples, 2))*self.scale_sigma).type(torch.DoubleTensor))
        # self.rotation_values = torch.nn.Parameter(torch.atanh(torch.zeros((num_samples, 1))).type(torch.DoubleTensor))

        self.Emin = Emin
        self.Emax = Emax
        self.penal = penal

        self.points_inside_elements = {}

    def get_x(self, save_kernel_sums=False):

        # TODO: rewrite becasu sclaing is used twice

        W_scale = self.W_scale[self.persistent_mask]
        W_sigmas_ratio = self.W_sigmas_ratio[self.persistent_mask] 
        W_rotation = self.W_rotation[self.persistent_mask]
        W_offsets = self.W_offsets[self.persistent_mask]
        batch_size = W_scale.shape[0]

        base_scale = (self.scale_max - self.scale_min)*torch.sigmoid(W_scale) + self.scale_min
        base_scale = base_scale.squeeze()
        sigmas_ratio = (self.sigmas_ratio_max - self.sigmas_ratio_min)*torch.sigmoid(W_sigmas_ratio) + self.sigmas_ratio_min    
        scale = torch.cat([torch.ones(batch_size, 1), torch.ones(batch_size, 1) * sigmas_ratio], dim=1)
        # scale = torch.cat([base_scale, base_scale * sigmas_ratio], dim=1)
        # rotation = np.pi/2 * torch.tanh(W_rotation).view(batch_size)
        rotation = self.rotation_min + (self.rotation_max - self.rotation_min)*torch.sigmoid(W_rotation).view(batch_size)
        # offsets = torch.tanh(W_offsets)*self.scaler
        offsets = self.coord_min + (self.coord_max - self.coord_min)*torch.sigmoid(W_offsets)

        if save_kernel_sums:
            # Calculate pairwise Euclidean distances between offsets
            n = offsets.shape[0]
            offsets_flat = offsets.squeeze()  # Remove extra dimensions to get (n,2) shape
            distances = torch.cdist(offsets_flat, offsets_flat, p=2)  # Compute pairwise L2 distances

        cos_rot = torch.cos(rotation)
        sin_rot = torch.sin(rotation)

        # rotation matrix
        R = torch.stack([
            torch.stack([cos_rot, -sin_rot], dim=-1),
            torch.stack([sin_rot, cos_rot], dim=-1)
        ], dim=-2)

        # S = torch.diag_embed(scale)

        # Compute covariance matrix: RSS^TR^T
        # pre_covariance = R @ S @ S @ R.transpose(-1, -2)
        # pre_covariance = S * S
        # print("det_coeffi: ", torch.sqrt(0.00005/torch.det(pre_covariance).sum()))
        # covariance = torch.sqrt(0.00005/torch.det(pre_covariance).sum())*pre_covariance
        # covariance = pre_covariance + torch.eye(2).unsqueeze(0).expand(batch_size, -1, -1) * 1e-6
        # print("covariance: ", covariance.shape)
        # print("base_scale: ", base_scale.shape)

        # covariance = covariance #* base_scale 
        # covariance_diag = torch.diagonal(covariance, dim1=-2, dim2=-1)
        covariance_diag = scale**2
        pre_inv_covariance_diag = 1/(covariance_diag + 1e-6)

        pre_inv_covariance = torch.diag_embed(pre_inv_covariance_diag)
        # pre_inv_covariance = pre_inv_covariance

        # Compute inverse covariance
        coords = self.coords.unsqueeze(0).expand(batch_size, -1, -1, -1)
        offsets = offsets[:, None, None, :]
        xy = coords - offsets

        xy = xy /((base_scale[:, None, None, None])+1e-8)

        # Apply rotation to coordinates
        rotated_xy = torch.einsum('bij,bxyj->bxyi', R, xy)
        
        # Calculate z using pre_inv_covariance (no rotation)
        z = torch.einsum('bxyi,bij,bxyj->bxy', rotated_xy, -0.5 * pre_inv_covariance, rotated_xy)
        # z = z + torch.log(torch.tensor(100))
        # kernel = torch.exp(z) / (2 * torch.tensor(np.pi) * torch.sqrt(torch.det(covariance)+1e-8)).view(batch_size, 1, 1)
        # Add eps to determinant calculations
        # det_covariance = torch.det(covariance) + 1e-8
        # det_covariance = torch.prod(covariance_diag, dim=1)
        # det_covariance = det_covariance * base_scale**4
        # Safeguard the sqrt operation
        # print("det_covariance: ", det_covariance.shape)
        # sqrt_det = torch.sqrt(torch.clamp(det_covariance, min=1e-8))
        # print("sqrt_det: ", sqrt_det.shape)
        # print("sqrt_det: ", sqrt_det)
        # print("area: ", 2 * torch.tensor(np.pi) * sqrt_det)
        # Modify kernel calculation to prevent division by zero
        # kernel = torch.exp(z) / (2 * torch.tensor(np.pi) * sqrt_det).view(batch_size, 1, 1)
        kernel = torch.exp(z)
        kernel = torch.squeeze(kernel)

        # print("kernel: ", kernel.min(), kernel.max(), kernel.shape)

        # TODO: find a more suitable threshold for merging (large overlap, volfrac convergence criteria, max_scale or max det)
        if save_kernel_sums:
            # kernel_sigmoids = torch.sigmoid(kernel) 
            # Compute pairwise sums of kernel vectors
            n = kernel.shape[0]  # number of vectors
            pairwise_sums = torch.zeros((n, n), device=kernel.device)

            close_gauss_pairs = []
            close_pair_kernel_sums = []
            self.points_inside_elements = {}

            for i in range(n):
                for j in range(n):
                    if j > i:
                        sum_ij = kernel[i] + kernel[j]
                        pairwise_sums[i,j] = sum_ij.max()
                        # print("sum_{}_{}: ".format(i, j), sum_ij.max())
                        if sum_ij.max() > 1.0:
                            close_gauss_pairs.append((i, j))
                            close_pair_kernel_sums.append(sum_ij.max())

                            if i not in self.points_inside_elements.keys():
                                self.points_inside_elements[i] = self.coords[kernel[i] > 0.5]

                            if j not in self.points_inside_elements.keys():
                                self.points_inside_elements[j] = self.coords[kernel[j] > 0.5]
            # print("Pairwise max sums shape:", pairwise_sums.shape)
            print(pairwise_sums)

            # Convert pairs and sums to tensors and sort by sum values
            if close_gauss_pairs:  # Only process if there are pairs
                pairs_tensor = torch.tensor(close_gauss_pairs, device=pairwise_sums.device)
                sums_tensor = torch.tensor(close_pair_kernel_sums, device=pairwise_sums.device)
                
                # Sort by kernel sums in descending order
                sorted_indices = torch.argsort(sums_tensor, descending=True)
                pairs_tensor = pairs_tensor[sorted_indices]
                sums_tensor = sums_tensor[sorted_indices]
                
                close_gauss_pairs = pairs_tensor.tolist()
                close_pair_kernel_sums = sums_tensor.tolist()

            self.merging_pairs = close_gauss_pairs
            self.merging_kernel_sums = close_pair_kernel_sums
             
        kernel_sum = kernel.sum(dim=0)+1e-8

        # print("kernel_sum: ", kernel_sum.min(), kernel_sum.max(), kernel_sum.shape)
        # kernel_sum_features = torch.log(kernel_sum)
        # print("kernel_sum_features: ", kernel_sum_features.min(), kernel_sum_features.max(), kernel_sum_features.shape)

        # self.H = (1 - self.Emin)*torch.sigmoid(-4*kernel_sum_features) + self.Emin
        self.H = (1 - self.Emin)*torch.sigmoid(-self.smooth_k*(kernel_sum - 0.4)) + self.Emin
        self.H_splitted = (1 - self.Emin)*torch.sigmoid(1.8*self.smooth_k*(kernel - 0.1)) + self.Emin
        # self.H_splitted = (1 - self.Emin)*torch.sigmoid(4*torch.log(kernel*1.1)) + self.Emin
        self.H_splitted_sum = self.H_splitted.sum(dim=0)
        # print("H_splitted: ", self.H_splitted.min(), self.H_splitted.max(), self.H_splitted.shape)
        # print("H_splitted_sum: ", self.H_splitted_sum.min(), self.H_splitted_sum.max(), self.H_splitted_sum.shape)
        self.H_offset = (1 - self.Emin)*torch.sigmoid(-4*torch.log(kernel_sum*1.1)) + self.Emin

        return self.H

    def prepare_grads(self):
        print("persistent_mask: ", self.persistent_mask)
        self.W_scale.grad.data[~self.persistent_mask] = 0.0
        self.W_sigmas_ratio.grad.data[~self.persistent_mask] = 0.0
        self.W_rotation.grad.data[~self.persistent_mask] = 0.0
        self.W_offsets.grad.data[~self.persistent_mask] = 0.0
        # self.W.grad.data[self.persistent_mask, 3:] *= 0.1

        # W_scale = self.W_scale
        # W_sigmas_ratio = self.W_sigmas_ratio 

        # base_scale = (self.scale_max - self.scale_min)*torch.sigmoid(W_scale) + self.scale_min
        # sigmas_ratio = (self.sigmas_ratio_max - self.sigmas_ratio_min)*torch.sigmoid(W_sigmas_ratio) + self.sigmas_ratio_min
        # scale = torch.cat([base_scale, base_scale * sigmas_ratio], dim=1)

        # # don't update sigmasgradients for small determinants
        # # scale = (self.sigma_max - self.sigma_min)*torch.sigmoid(self.W_scale_rotation[:, 0:2]) + self.sigma_min
        # S = torch.diag_embed(scale)
        # covariance = S @ S.transpose(-1, -2)
        # det_covariance = torch.det(covariance)

        # print("#####   prepare_grads   #####")
        # print("det_covariance: ", det_covariance[self.persistent_mask])
        # valid_det_mask = det_covariance < self.det_min
        # scale1_gradient_mask = torch.logical_and(valid_det_mask, (self.W_scale.grad.data[:, 0] > 0))    
        # # scale2_gradient_mask = torch.logical_and(valid_det_mask, (self.W_scale_rotation.grad.data[:, 1] > 0))

        # print("scale1_grad: ", self.W_scale.grad.data[self.persistent_mask, 0])
        # print("scale2_grad: ", self.W_sigmas_ratio.grad.data[self.persistent_mask, 0])
        # print("scale1_gradient_mask: ", scale1_gradient_mask[self.persistent_mask])
        # # print("scale2_gradient_mask: ", scale2_gradient_mask[self.persistent_mask])

        # self.W_scale.grad.data[scale1_gradient_mask, 0] = 0.0
        # # self.W_scale_rotation.grad.data[scale2_gradient_mask, 1] = 0.0

        # print("Determinant sum: ", det_covariance[self.persistent_mask].sum())

    def forward(self, ce):

        # compliance_w = 2*(10**4)
        # volfrac_w = 1 
        # gaussian_overlap_w = 2

        # Add numerical stability to compliance calculation
        H_vec = torch.clamp(self.H, min=self.Emin, max=self.Emax)**self.penal
        compliance = torch.dot(H_vec, ce)
        
        # Safeguard the loss calculations
        volfrac_loss_pre = torch.nn.functional.relu(
            self.H.mean() - 0.6
        )
        
        gaussian_overlap = torch.nn.functional.relu(
            self.H_splitted_sum - 1.5
        ).mean()

        print("volume: ", self.H.mean())
        print("compliance: ", compliance)
        print("gaussian_overlap: ", gaussian_overlap)
        # print("comb_loss: ", 0.00001*self.combine_loss.sum())

        # if volfrac_loss > 0.6:
        # obj_ce -= 0.00000001*volfrac_loss
        # obj_ce = 0.0000001*self.combine_loss.sum()

        # obj_ce = compliance/(volfrac_loss)
        # obj_ce = (volfrac_loss-1)/(2*(1-compliance))
        # obj_ce = compliance


        obj_ce = -compliance*self.compliance_w + volfrac_loss_pre*self.volfrac_w + gaussian_overlap*self.gaussian_overlap_w
        obj_real = compliance*self.compliance_w + volfrac_loss_pre*self.volfrac_w + gaussian_overlap*self.gaussian_overlap_w

        return obj_ce, volfrac_loss_pre, gaussian_overlap, compliance, obj_real
    
    def splitting_1(self):
        # find ids for splitting
        W_scale_rotation = self.W_scale_rotation[self.persistent_mask]
        W_offsets = self.W_offsets[self.persistent_mask]
        batch_size = W_scale_rotation.shape[0]

        scale = (self.sigma_max - self.sigma_min)*torch.sigmoid(W_scale_rotation[:, 0:2]) + self.sigma_min
        rotation = np.pi/2 * torch.tanh(W_scale_rotation[:, 2]).view(batch_size)
        offsets = torch.tanh(W_offsets)*self.scaler

        sigmas_maxs = torch.amax(scale, dim=1)
        sigmas_mins = torch.amin(scale, dim=1)
        sigmas_ratio = sigmas_maxs/sigmas_mins

        splitting_indices = torch.argsort(sigmas_ratio)[-2:]

        start_index = self.current_marker + 1
        end_index = self.current_marker + 1 + len(splitting_indices)
        self.persistent_mask[start_index: end_index] = True
        self.W_scale_rotation.data[start_index:end_index, :] = self.W_scale_rotation.data[splitting_indices, :]
        self.W_offsets.data[start_index:end_index, :] = self.W_offsets.data[splitting_indices, :]
        # scale_reduction_factor = 2**(1/8)
        scale_reduction_factor = 1/(2**0.25)
        scale_new = (self.sigma_max - self.sigma_min)*torch.sigmoid(self.W_scale_rotation.data[start_index:end_index, 0:2]) + self.sigma_min
        scale_new *= scale_reduction_factor
        scale_new = (scale_new - self.sigma_min)/(self.sigma_max - self.sigma_min)

        self.W_scale_rotation.data[start_index:end_index, 0:2] = torch.logit(scale_new)
        self.W_scale_rotation.data[splitting_indices, 0:2] = torch.logit(scale_new)
        self.current_marker = self.current_marker + len(splitting_indices)

    def cleaning(self):
        # scale = (self.sigma_max - self.sigma_min)*torch.sigmoid(self.W.data[:, 0:2]) + self.sigma_min


        # print("Scale stat for cleaning: ")
        # print(scale.min(), scale.mean(), scale.max())

        W_scale_rotation = self.W_scale_rotation[self.persistent_mask]
        W_offsets = self.W_offsets[self.persistent_mask]
        batch_size = W_scale_rotation.shape[0]

        scale = (self.sigma_max - self.sigma_min)*torch.sigmoid(W_scale_rotation[:, 0:2]) + self.sigma_min
        rotation = np.pi/2 * torch.tanh(W_scale_rotation[:, 2]).view(batch_size)
        offsets = torch.tanh(W_offsets)*self.scaler

        # print("Sigmas: ", scale)
        # offsets = torch.tanh(self.offsets_values)*self.scaler
        # offsets[:, 1] = self.width_ratio*offsets[:, 1]
        # offsets = self.offsets_values
        # batch_size = offsets.shape[0]
        # rotation = np.pi/2 * torch.tanh(self.rotation_values).view(batch_size)

        # Compute the components of the covariance matrix
        cos_rot = torch.cos(rotation)
        sin_rot = torch.sin(rotation)

        R = torch.stack([
            torch.stack([cos_rot, -sin_rot], dim=-1),
            torch.stack([sin_rot, cos_rot], dim=-1)
        ], dim=-2)

        S = torch.diag_embed(scale)

        # Compute covariance matrix: RSS^TR^T
        pre_covariance = R @ S @ S @ R.transpose(-1, -2)
        det_coeff = torch.sqrt(0.00005/torch.det(pre_covariance).sum())
        
        corrected_scale = torch.sqrt(det_coeff)*scale
        corrected_scale_min = torch.amin(corrected_scale, dim=1)

        good_scale_mask = corrected_scale_min > 0.018
        persistent_mask_indices = torch.arange(self.persistent_mask.shape[0])[self.persistent_mask]
        self.persistent_mask[persistent_mask_indices] = good_scale_mask 


        # print("Scale stat for cleaning: ")
        # print(corrected_scale.min(), corrected_scale.mean(), corrected_scale.max())
                
    def splitting_2(self):
        # find ids for splitting
        # output = self.W[self.persistent_mask]
        # batch_size = output.shape[0]

        # scale = (self.sigma_max - self.sigma_min)*torch.sigmoid(output[:, 0:2]) + self.sigma_min
        # # print("sigmas_1", scale)
        # rotation = np.pi/2 * torch.tanh(output[:, 2]).view(batch_size)
        # offsets = torch.tanh(output[:, 3:])*self.scaler

        # sigmas_maxs = torch.amax(scale, dim=1)
        # sigmas_mins = torch.amin(scale, dim=1)
        # sigmas_ratio = sigmas_maxs/sigmas_mins

        # splitting_indices = torch.argsort(sigmas_ratio)[-2:]

        grad_threshold = 3e-6
        gauss_threshold = 0.072
        sigmas_r_threshold = 1.1
        sigma_min_threshold = 0.0035
        top_k_split = 2

        # gradient_norms = torch.norm(self.W.grad[self.persistent_mask][:, 3:], dim=1, p=2)
        # gaussian_norms = torch.norm(torch.sigmoid(self.W.data[self.persistent_mask][:, 0:2]), dim=1, p=2)



        scale = (self.sigma_max - self.sigma_min)*torch.sigmoid(self.W_scale_rotation[:, 0:2]) + self.sigma_min
        S = torch.diag_embed(scale)
        covariance = S @ S.transpose(-1, -2)

        sigmas_maxs = torch.amax(scale, dim=1)
        sigmas_mins = torch.amin(scale, dim=1)
        sigmas_ratio = sigmas_maxs/sigmas_mins
        covariance_det = torch.det(covariance)

        print("Sigmas_ratio: ", sigmas_ratio)
        print("covariance det: ", covariance_det)
        print("covariance_det_sum: ", torch.sum(covariance_det))

        # Sort indices by determinant in descending order
        sorted_det, sorted_det_indices = torch.sort(covariance_det, descending=True)

        # Take top k indices with largest determinants
        splitting_indices = sorted_det_indices[sorted_det > self.det_max][:top_k_split]

        # Ensure we have unique indices
        splitting_indices = torch.unique(splitting_indices)
        # splitting_indices = torch.empty(0, dtype=torch.long)
        
        # print("large_sigmas_r_indices: ", large_sigmas_r_indices)
        print("Split ids: ", splitting_indices)
        print("covariance det after: ", covariance_det[splitting_indices])
        # common_indices = large_gradient_indices[common_indices_mask]

        # splitting_indices = large_gradient_indices[common_indices_mask]

        #########################
        
        # scale_reduction_factor = 2**(1/8)
        scale_reduction_factor = 1/(2**0.25)
        scale_pre_init = (self.sigma_max - self.sigma_min)*torch.sigmoid(self.W_scale_rotation.data[splitting_indices, 0:2]) + self.sigma_min
        # main_axis_longer = (scale_pre[:, 0] / scale_pre[:, 1]) >= 1
        scale_pre_init *= scale_reduction_factor
        scale_new_pre = torch.sqrt(scale_pre_init[:, 0]*scale_pre_init[:, 1])
        scale_new = scale_pre_init
        scale_new[:, 0] = scale_new_pre
        scale_new[:, 1] = scale_new_pre
        
        scale_new = torch.logit((scale_new - self.sigma_min)/(self.sigma_max - self.sigma_min))
        
        start_index = self.current_marker + 1
        end_index = self.current_marker + 1 + len(splitting_indices)
        self.persistent_mask[start_index: end_index] = True
        self.W_scale_rotation.data[start_index:end_index, :] = self.W_scale_rotation.data[splitting_indices, :]
        self.W_offsets.data[start_index:end_index, :] = self.W_offsets.data[splitting_indices, :]

        self.W_scale_rotation.data[start_index:end_index, 0:2] = scale_new
        self.W_scale_rotation.data[splitting_indices, 0:2] = scale_new

        rotation = np.pi/2 * torch.tanh(self.W_scale_rotation.data[start_index:end_index, 2])
        offsets = torch.tanh(self.W_offsets.data[start_index:end_index])*self.scaler

        # rotation_new = rotation
        # sigmas = (self.sigma_max - self.sigma_min)*torch.sigmoid(self.W.data[start_index:end_index, 0:2]) + self.sigma_min
        main_axis_longer = (scale_pre_init[:, 0] / scale_pre_init[:, 1]) >= 1
        moving_vectors = torch.zeros_like(scale_pre_init)

        main_axis_vec = torch.vstack([torch.cos(rotation), torch.sin(rotation)]).T
        minor_axis_vec = torch.vstack([torch.sin(rotation), -torch.cos(rotation)]).T

        moving_vectors[main_axis_longer] = main_axis_vec[main_axis_longer]
        moving_vectors[~main_axis_longer] = minor_axis_vec[~main_axis_longer]

        scale_new_pre = scale_new_pre[:, None]

        offsets_new_forw = offsets + moving_vectors*scale_new_pre*1
        offsets_new_back = offsets - moving_vectors*scale_new_pre*1

        # offsets_new = offsets

        # rotation_w = torch.atan(rotation_new*2/np.pi)
        offsets_forw_w = torch.atan(offsets_new_forw/self.scaler)
        offsets_back_w = torch.atan(offsets_new_back/self.scaler)

        # self.W_scale_rotation.data[start_index:end_index, 2] = 0
        # self.W_scale_rotation.data[splitting_indices, 2] = 0

        # self.W_offsets.data[start_index:end_index] = offsets_forw_w
        # self.W_offsets.data[splitting_indices] = offsets_back_w

        self.current_marker = self.current_marker + len(splitting_indices)

    def merging(self):
        # TODO: rotation is inncorrectly used (need to take into account sin). 
        # also check the result areas
        # check determinant before splitting
        W_scale = self.W_scale[self.persistent_mask]
        W_sigmas_ratio = self.W_sigmas_ratio[self.persistent_mask]
        batch_size = W_scale.shape[0]

        base_scale = (self.scale_max - self.scale_min)*torch.sigmoid(W_scale) + self.scale_min
        sigmas_ratio = (self.sigmas_ratio_max - self.sigmas_ratio_min)*torch.sigmoid(W_sigmas_ratio) + self.sigmas_ratio_min
        # print("sigmas_ratio: ", sigmas_ratio.shape)
        # print("base_scale: ", base_scale.shape)
        scale = torch.cat([base_scale, base_scale * sigmas_ratio], dim=1)
        
        covariance_diag = scale.pow(2)
        det_covariance = torch.prod(covariance_diag, dim=1)
        # print("det_covariance: ", det_covariance)
        # print("det_covariance_sum: ", det_covariance.sum())

        ###############################################################################################################
        persistent_mask_indices = torch.arange(self.persistent_mask.shape[0])[self.persistent_mask]

        areas = np.pi * np.sqrt(det_covariance.detach().cpu().numpy())

        # print("merging pairs: ", self.merging_pairs)

        while len(self.merging_pairs) > 0:
            process_pair = self.merging_pairs.pop(0)
            print(process_pair)
            # new_offset = (offsets[process_pair[0]] + offsets[process_pair[1]])/

            current_index = self.current_marker + 1

            # print("points_inside_elements: ", self.points_inside_elements.keys())

            x = torch.cat([self.points_inside_elements[process_pair[0]], self.points_inside_elements[process_pair[1]]])
            x = x.unique(dim=0)
            x_mean = x.mean(dim=0).clone()
            x = x - x_mean
            x = x.detach().cpu().numpy()

            # print("x.shape: ", x.shape)
            # print("x: ", x.min(), x.max())

            target_area = areas[process_pair[0]] + areas[process_pair[1]]

            center, R, Sigma = fit_ellipsoid(x, target_area)
            center = x_mean - torch.tensor(center)
            R = torch.tensor(R)
            # center = torch.tensor(x).mean(dim=0)

            # print("R: ", R)

            # alpha = torch.abs(torch.tensor(np.arctan2(R[1, 0], R[0, 0])))
            alpha = torch.arccos(R[0, 0])
            # alpha = torch.tensor(np.arctan2(R[1, 0], R[0, 0]))
            if alpha > torch.pi/2:
                alpha = alpha - torch.pi
            elif alpha < -torch.pi/2:
                alpha = alpha + torch.pi

            # print("alpha: ", alpha)
            # print("atan_alpha: ", torch.atan(alpha*2/np.pi))
            scale = torch.tensor(Sigma[0])
            ratio = torch.tensor(Sigma[1]/Sigma[0])

            # result_area = np.pi * np.sqrt(np.linalg.det(np.diag(Sigma) @ np.diag(Sigma).T))
            scale = min(
                self.scale_max - 0.1 * (self.scale_max - self.scale_min),
                max(self.scale_min + 0.1 * (self.scale_max - self.scale_min), scale)
            )
            ratio = min(
                self.sigmas_ratio_max - 0.1 * (self.sigmas_ratio_max - self.sigmas_ratio_min),
                max(self.sigmas_ratio_min + 0.1 * (self.sigmas_ratio_max - self.sigmas_ratio_min), ratio)
            )

            scale = torch.tensor(scale)
            ratio = torch.tensor(ratio)

            # print("target_area:", target_area)
            # print("result_area: ", result_area)
            # print("scale_range: ", self.scale_max, self.scale_min)
            # print("scale: ", scale)
            # print("logit_scale: ", torch.logit((scale - self.scale_min)/(self.scale_max - self.scale_min)))
            # print("ratio_range: ", self.sigmas_ratio_max, self.sigmas_ratio_min)
            # print("ratio: ", ratio)
            # print("logit_ratio: ", torch.logit((ratio - self.sigmas_ratio_min)/(self.sigmas_ratio_max - self.sigmas_ratio_min)))

            # print("center_range: ", self.coord_min, self.coord_max)
            # print("center: ", center)
            # print("logit_center: ", torch.logit((center - self.coord_min)/(self.coord_max - self.coord_min)))

            self.W_rotation.data[current_index] = torch.logit((alpha - self.rotation_min)/(self.rotation_max - self.rotation_min))
            self.W_scale.data[current_index] = torch.logit((scale - self.scale_min)/(self.scale_max - self.scale_min))
            self.W_sigmas_ratio.data[current_index] = torch.logit((ratio - self.sigmas_ratio_min)/(self.sigmas_ratio_max - self.sigmas_ratio_min))
            self.W_offsets.data[current_index] = torch.logit((center - self.coord_min)/(self.coord_max - self.coord_min))

            self.persistent_mask[current_index] = True
            self.persistent_mask[persistent_mask_indices[process_pair[0]]] = False
            self.persistent_mask[persistent_mask_indices[process_pair[1]]] = False

            # Remove any remaining pairs that contain the merged indices
            self.merging_pairs = [pair for pair in self.merging_pairs if process_pair[0] not in pair and process_pair[1] not in pair]

            self.current_marker += 1
        ##########################################################################################################
        # check determinant before splitting
        W_scale = self.W_scale[self.persistent_mask]
        W_sigmas_ratio = self.W_sigmas_ratio[self.persistent_mask]
        batch_size = W_scale.shape[0]

        base_scale = (self.scale_max - self.scale_min)*torch.sigmoid(W_scale) + self.scale_min
        sigmas_ratio = (self.sigmas_ratio_max - self.sigmas_ratio_min)*torch.sigmoid(W_sigmas_ratio) + self.sigmas_ratio_min    
        scale = torch.cat([base_scale, base_scale * sigmas_ratio], dim=1)
        
        covariance_diag = scale.pow(2)
        det_covariance = torch.prod(covariance_diag, dim=1)
        # print("det_covariance: ", det_covariance)
        # print("det_covariance_sum: ", det_covariance.sum())

        ###############################################################################################################
    
    def densification(self):

        # check determinant before splitting
        output = self.W_scale_rotation[self.persistent_mask]
        batch_size = output.shape[0]
        scale = (self.sigma_max - self.sigma_min)*torch.sigmoid(output[:, 0:2]) + self.sigma_min
        rotation = np.pi/2 * torch.tanh(output[:, 2]).view(batch_size)
        cos_rot = torch.cos(rotation)
        sin_rot = torch.sin(rotation)
        R = torch.stack([
            torch.stack([cos_rot, -sin_rot], dim=-1),
            torch.stack([sin_rot, cos_rot], dim=-1)
        ], dim=-2)
        S = torch.diag_embed(scale)
        # Compute covariance matrix: RSS^TR^T
        pre_covariance = R @ S @ S @ R.transpose(-1, -2)
        covariance = torch.sqrt(0.00005/torch.det(pre_covariance).sum())*pre_covariance
        # print("precov_DET: ", torch.det(pre_covariance))
        # print("precov_DET_half: ", torch.det(pre_covariance)/2)
        # print("DETERMINanat list before: ", torch.det(covariance))

        ###############################################################################################################

        # Calculate the norm of gradients
        # scale = (self.sigma_max - self.sigma_min)*torch.sigmoid(output[:, 0:2]) + self.sigma_min
        # sigmas_maxs = torch.amax(scale, dim=1)
        # sigmas_mins = torch.amin(scale, dim=1)
        # sigmas_ratio = sigmas_maxs/sigmas_mins
        # gradient_norms = torch.norm(self.W.grad[self.persistent_mask][:, 3:], dim=1, p=2)
        # gaussian_norms = torch.norm(torch.sigmoid(self.W.data[self.persistent_mask][:, 0:2]), dim=1, p=2)

        # print("######################### stats for criteria ###################################")
        # print("gradient")
        # print(gradient_norms.min(), gradient_norms.mean(), gradient_norms.max())
        # print("gauss")
        # print(gaussian_norms.min(), gaussian_norms.mean(), gaussian_norms.max())
        # print("scale ratio")
        # print(sigmas_ratio.min(), sigmas_ratio.mean(), sigmas_ratio.max())

        # print("################################################################################")

        # find ids for splitting
        self.splitting_2()

        ##########################################################################################################
        # check determinant before splitting
        output = self.W_scale_rotation[self.persistent_mask]
        batch_size = output.shape[0]
        scale = (self.sigma_max - self.sigma_min)*torch.sigmoid(output[:, 0:2]) + self.sigma_min
        rotation = np.pi/2 * torch.tanh(output[:, 2]).view(batch_size)
        cos_rot = torch.cos(rotation)
        sin_rot = torch.sin(rotation)
        R = torch.stack([
            torch.stack([cos_rot, -sin_rot], dim=-1),
            torch.stack([sin_rot, cos_rot], dim=-1)
        ], dim=-2)
        S = torch.diag_embed(scale)
        # Compute covariance matrix: RSS^TR^T
        pre_covariance = R @ S @ S @ R.transpose(-1, -2)
        covariance = torch.sqrt(0.00005/torch.det(pre_covariance).sum())*pre_covariance
        print("precov_DET: ", torch.det(pre_covariance))
        print("DETERMINanat list after: ", torch.det(covariance))

        ###############################################################################################################

    def get_final_x(self):

        W_scale = self.W_scale[self.persistent_mask]
        W_sigmas_ratio = self.W_sigmas_ratio[self.persistent_mask] 
        W_rotation = self.W_rotation[self.persistent_mask]
        W_offsets = self.W_offsets[self.persistent_mask]
        batch_size = W_scale.shape[0]

        base_scale = (self.scale_max - self.scale_min)*torch.sigmoid(W_scale) + self.scale_min
        base_scale = base_scale.squeeze()
        sigmas_ratio = (self.sigmas_ratio_max - self.sigmas_ratio_min)*torch.sigmoid(W_sigmas_ratio) + self.sigmas_ratio_min    
        scale = torch.cat([torch.ones(batch_size, 1), torch.ones(batch_size, 1) * sigmas_ratio], dim=1)
        rotation = self.rotation_min + (self.rotation_max - self.rotation_min)*torch.sigmoid(W_rotation).view(batch_size)
        offsets = self.coord_min + (self.coord_max - self.coord_min)*torch.sigmoid(W_offsets)

        design_vars = {"scale": base_scale.detach().cpu().numpy().tolist(),
                       "sigmas_ratio": sigmas_ratio.detach().cpu().numpy().tolist(),
                       "rotation": rotation.detach().cpu().numpy().tolist(),
                       "offsets": offsets.detach().cpu().numpy().tolist()}

        return design_vars