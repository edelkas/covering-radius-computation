import numpy as num
import itertools as itt
import scipy as sci
import sympy as sym
import time as tm
from scipy.optimize import linprog


def in_unit_cube(p):
    """
        p a point of dimension n
        returns whether p is in [0,1]^n
    """

    array_p = num.array(list(p))
    len_p = len(list(p))
    
    return ((array_p >= num.zeros((len_p,1))).all() and (array_p <= num.ones((len_p,1))).all())

    
def in_interior_of_translate(p,mu,P,N):
    """
        p a point, mu a scalar, P a polytope, N a finite set of lattice points
        returns whether p is contained in the interior of mu*P + z, for some z in N
    """
    
    P_scaled = mu*P

    for z in N:
        Q = P_scaled.minkowski_sum(Polyhedron(vertices=[list(z)]))
        if Q.interior_contains(list(vector(p))):
            return true
    return false

    
def contains_origin_as_int(P):
    """
        P a polytope
        returns whether P contains the origin in its interior
    """

    return min(fac[0] for fac in P.inequalities_list()) > 0
    

def move_origin(P):
    """
        P a polytope
        translates P as to contain the origin in its interior (if necessary)
    """

    if contains_origin_as_int(P):
        return P
    return P.minkowski_sum(Polyhedron(vertices=[list(-P.centroid())]))

    
def mu_bound(P):
    """
        P a polytope
        determines an upper bound on the covering radius of P
    """
    
    # determine minimal scalar 'alpha' such that alpha*P is a lattice polytope
    lat_pol = P.is_lattice_polytope()
    vertices_P = P.vertices_list()
    dim_P = P.dim()
    if lat_pol:
        alpha = 1/gcd([gcd(v) for v in vertices_P])
    else:
        alpha = lcm([lcm([Rational(r).denominator() for r in v]) for v in vertices_P])

    # method 1: lattice polytope bound
    mu1 = alpha*dim_P
    
    # method 2: for dimension 2 the interior lattice point number bound of Codenotti, Santos & Schymura
    if dim_P == 2:
        num_ints = len(LatticePolytope((alpha*P).vertices_list()).interior_points())
        if num_ints > 0:
            mu2 = alpha*(1/2+1/(num_ints+1))
        else:
            mu2 = mu1
    else:
        mu2 = mu1
    
    # method 3: for dimension 3 and alpha*P non-hollow the bound 3/2 of Codenotti, Santos & Schymura
    if dim_P == 3 and len(LatticePolytope((alpha*P).vertices_list()).interior_points()) > 0:
        mu3 = alpha*3/2
    else:
        mu3 = mu1

    # method 4: via flatness theorem and computation of lattice-width
    # not yet implemented
    
    mu0 = min(mu1,mu2,mu3)

    return mu0


def inf_norm(P):
    """
        P a polytope
        determines infinity norm of P [equals beta(P) in Cslovjecsek, Malikiosis, Naszodi & Schymura]
    """

    return max([max([abs(v[i]) for i in range(P.dim())]) for v in P.vertices_list()])
    

def neighbors(P,mu0):
    """
        P a rational polyhedron, mu0 an upper bound on the covering radius of P
        returns a finite (super)set of the lattice points that are involved in covering the unit cube by translates of mu(P)*P
        [is the set N_P in Cslovjecsek, Malikiosis, Naszodi & Schymura]
    """

    C = polytopes.hypercube(P.dim(),intervals = 'zero_one')
    Q = Polyhedron(vertices=(-mu0*P).vertices_list())
    
    return (C.minkowski_sum(Q)).integral_points()   # returns a tuple of tuples
    

def covering_radii(L):
    """
        L a list of lists of vertices (each corresponding to a polytope)
        returns the list L amended with the covering radii (and a last-covered point) of the contained polytopes
    """

    extendedL = []
    
    for elem in L:
      P = Polyhedron(vertices=elem)
      res = covering_radius(P)
      extendedL.append([elem,res[0],res[1]])
    
    return extendedL

    
def covering_radius(P):
    """
        P a rational polyhedron
        returns the covering radius of P and a last-covered point
    """

    # setting up the problem
    mu_max = 0
    mu0 = mu_bound(P)
    res_test = []     # efficiency list: maintains the solutions (mu,p) that don't pass the in_interior_of_translate test in order to avoid multiple superfluous candidate checking
    p_last = []
    n = P.dim()
    P = move_origin(P)
    N_P = neighbors(P,mu0)
    
    for facet_normals in itt.combinations(P.inequalities_list(),n+1):

      # test for linearly independent choice
      if Matrix(QQ,list(facet_normals)).rank() < n+1:
        continue

      # LHS = facet_normals in form [b -A] for linear system to be solved in upcoming loop (independent of the loop variable)
      lhs = Matrix(QQ,list(facet_normals))
      lhs_inv = lhs.inverse()
      rhs_a = lhs[:,1:n+1]      # equals -A
  
      for anchor_points in itt.product(N_P,repeat=n+1): # takes (n+1)-tuples with repetition out of neighbors(P)

        # build the system of linear equations that is to be solved
                      
        # RHS = -A[i]*anchor_points[i]
        rhs_z = Matrix(QQ,anchor_points).transpose()
        rhs_list = []
        for i in range(n+1):
          rhs_list.append(rhs_a[i]*rhs_z[:,i])
        rhs = Matrix(QQ,rhs_list)       
      
        # solve the system exactly and find the unique solution
        result = vector(lhs_inv*rhs)
        mu = result[0]
        p = result[1:n+1]

        # test whether solution is relevant
        if (mu < 0) or (result in res_test) or (mu <= mu_max) or (not in_unit_cube(p)):
          continue

        if not in_interior_of_translate(p,mu,P,N_P):
          mu_max = mu
          p_last = p
        else:
          res_test.append(result)

    return [mu_max,list(p_last)]
