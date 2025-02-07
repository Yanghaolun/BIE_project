from kifmm2d.scalar.fmm import FMM as KI_FMM
from kifmm2d.misc.utils import random2
import numpy as np
import numba
import time
import os

"""
Demonstration of the FMM for the Laplace Kernel

If N <= 50000, will do a direct sum and compare to this
Otherwise, will try to call FMMLIB2D through pyfmmlib2d
To compare to
If this fails, no comparison for correctness!

On my macbook pro N=50,000 takes the direct method ~7s, the FMM <1s
(with N_equiv=64, N_cutoff=500)
And gives error <5e-14
"""

cpu_num = int(os.cpu_count()/2)

# Laplace Kernel
@numba.njit(fastmath=True)
def Eval(sx, sy, tx, ty):
    dx = tx-sx
    dy = ty-sy
    d2 = dx*dx + dy*dy
    return -np.log(d2)/(4*np.pi)

# Laplace Gradient Kernel
@numba.njit(fastmath=True)
def Gradient_Eval1(sx, sy, tx, ty, tau, extras):
    dx = tx-sx
    dy = ty-sy
    d2 = dx*dx + dy*dy
    id2 = 1.0/d2
    scale = -1.0/(2*np.pi)
    u = -np.log(d2)/(4*np.pi)*tau
    ux = scale*dx*id2*tau
    uy = scale*dy*id2*tau
    return u, ux, uy

# Laplace Gradient Kernel
@numba.njit(fastmath=True)
def Gradient_Eval2(sx, sy, tx, ty, tau, extras):
    dx = tx-sx
    dy = ty-sy
    d2 = dx*dx + dy*dy
    id2 = 1.0/d2
    scale = -1.0/(2*np.pi)
    u = -np.log(d2)/(4*np.pi)*tau[0]
    ux = scale*dx*id2*tau[0]
    uy = scale*dy*id2*tau[0]
    return u, ux, uy

N_source = 10*100
N_target = 10*100
test = 'uniform' # clustered or circle or uniform
reference_precision = 4

# construct some data to run FMM on
if test == 'uniform':
    px = np.random.rand(N_source)
    py = np.random.rand(N_source)
    rx = np.random.rand(N_target)
    ry = np.random.rand(N_target)
    bbox = None
elif test == 'clustered':
    N_clusters = 10
    N_per_cluster = int((N_source / N_clusters))
    N_random = N_source - N_clusters*N_per_cluster
    center_clusters_x, center_clusters_y = random2(N_clusters, -99, 99)
    px, py = random2(N_source, -1, 1)
    px[:N_random] *= 100
    py[:N_random] *= 100
    px[N_random:] += np.repeat(center_clusters_x, N_per_cluster)
    py[N_random:] += np.repeat(center_clusters_y, N_per_cluster)
    px /= 100
    py /= 100
    rx = np.random.rand(N_target)
    ry = np.random.rand(N_target)
    bbox = [0,1,0,1]
elif test == 'circle':
    rand_theta = np.random.rand(int(N_source))*2*np.pi
    px = np.cos(rand_theta)
    py = np.sin(rand_theta)
    rx = (np.random.rand(N_target)-0.5)*10
    ry = (np.random.rand(N_target)-0.5)*10
    bbox = [-5,5,-5,5]
else:
    raise Exception('Test is not defined')

# maximum number of points in each leaf of tree for FMM
N_cutoff = 25
# number of modes in source/check surfaces
Nequiv = 30

# get random density
# tau = (np.random.rand(N_source))
tau = np.ones(N_source)
print('\nLaplace FMM with', N_source, 'source pts and', N_target, 'target pts.')

# get reference solution
reference = False
if reference:
    import pyfmmlib2d
    source = np.row_stack([px, py])
    target = np.row_stack([rx, ry])
    dumb_targ = np.row_stack([np.array([0.6, 0.6]), np.array([0.5, 0.5])])
    st = time.time()
    out = pyfmmlib2d.RFMM(source, dumb_targ, charge=tau, compute_target_potential=True, precision=reference_precision)
    tform = time.time() - st
    print('FMMLIB generation took:               {:0.1f}'.format(tform*1000))
    print('...Points/Second/Core (thousands)    \033[1m', int(N_source/tform/cpu_num/1000), '\033[0m ')
    st = time.time()
    out = pyfmmlib2d.RFMM(source, charge=tau, compute_source_potential=True, compute_source_gradient=True, precision=reference_precision)
    self_reference_eval = -0.5*out['source']['u']/np.pi
    self_reference_eval_x = -0.5*out['source']['u_x']/np.pi
    self_reference_eval_y = -0.5*out['source']['u_y']/np.pi
    tt = time.time() - st - tform
    print('FMMLIB self pot/grad eval took:       {:0.1f}'.format(tt*1000))
    print('...Points/Second/Core (thousands)    \033[1m', int(N_source/tt/cpu_num/1000), '\033[0m ')
    st = time.time()
    out = pyfmmlib2d.RFMM(source, target, charge=tau, compute_target_potential=True, compute_target_gradient=True, precision=reference_precision)
    target_reference_eval   = -0.5*out['target']['u']/np.pi
    target_reference_eval_x = -0.5*out['target']['u_x']/np.pi
    target_reference_eval_y = -0.5*out['target']['u_y']/np.pi
    tt = time.time() - st - tform
    print('FMMLIB target pot/grad eval took:     {:0.1f}'.format(tt*1000))
    print('...Points/Second/Core (thousands)    \033[1m', int(N_target/tt/cpu_num/1000), '\033[0m ')

FMM = KI_FMM(px, py, Eval, N_cutoff, Nequiv)
FMM.build_expansions(tau)
_ = FMM.source_evaluation(px[:20*N_cutoff], py[:20*N_cutoff])
_ = FMM.target_evaluation(rx[:20*N_cutoff], ry[:20*N_cutoff])
FMM.register_evaluator(Gradient_Eval1, Gradient_Eval2, 'gradient', 3)
_ = FMM.evaluate_to_points(px[:20*N_cutoff], py[:20*N_cutoff], 'gradient', True)
_ = FMM.evaluate_to_points(rx[:20*N_cutoff], ry[:20*N_cutoff], 'gradient', False)

output1 = FMM.evaluate_to_points(rx, ry, 'gradient', True)
print(output1.shape)
print(output1[:, :5])

def f(x, y, px, py):
    denom = 2 * np.pi * ((x - px)**2 + (y - py)**2)
    return np.sum((x - px) / denom), np.sum((y - py) / denom)

A = np.zeros((2, 5))
for i in range(5):
    A[:, i] = np.array(f(rx[i], ry[i], px, py))
print(output1[1:, :5] / A)


# print('')
# st = time.time()
# FMM = KI_FMM(px, py, Eval, N_cutoff, Nequiv, bbox=bbox, helper=FMM.helper)
# print('flexmm2d precompute took:             {:0.1f}'.format((time.time()-st)*1000))
# st = time.time()
# FMM.build_expansions(tau)
# tt = (time.time()-st)
# print('flexmm2d generation took:             {:0.1f}'.format(tt*1000))
# print('...Points/Second/Core (thousands)    \033[1m', int(N_source/tt/cpu_num/1000), '\033[0m ')
# st = time.time()
# self_fmm_eval = FMM.evaluate_to_points(px, py, 'gradient', True)
# tt = (time.time()-st)
# print('flexmm2d source eval took:            {:0.1f}'.format(tt*1000))
# print('...Points/Second/Core (thousands)    \033[1m', int(N_source/tt/cpu_num/1000), '\033[0m ')
# st = time.time()
# target_fmm_eval = FMM.evaluate_to_points(rx, ry, 'gradient', False)
# tt = (time.time()-st)
# print('flexmm2d target pot/grad eval took:   {:0.1f}'.format(tt*1000))
# print('...Points/Second/Core (thousands)    \033[1m', int(N_target/tt/cpu_num/1000), '\033[0m ')

if reference:
    scale = np.abs(self_reference_eval).max()
    self_err = np.abs(self_fmm_eval[0] - self_reference_eval)/scale
    self_err_x = np.abs(self_fmm_eval[1] - self_reference_eval_x)/scale
    self_err_y = np.abs(self_fmm_eval[2] - self_reference_eval_y)/scale
    self_err_grad = max(self_err_x.max(), self_err_y.max())
    target_err = np.abs(target_fmm_eval[0] - target_reference_eval)/scale
    target_err_x = np.abs(target_fmm_eval[1] - target_reference_eval_x)/scale
    target_err_y = np.abs(target_fmm_eval[2] - target_reference_eval_y)/scale
    target_err_grad = max(target_err_x.max(), target_err_y.max())
    print('\nMaximum difference, self:             {:0.2e}'.format(self_err.max()))
    print('Maximum difference, self-gradient:    {:0.2e}'.format(self_err_grad.max()))
    print('Maximum difference, target:           {:0.2e}'.format(target_err.max()))
    print('Maximum difference, targ-gradient:    {:0.2e}'.format(target_err_grad.max()))

