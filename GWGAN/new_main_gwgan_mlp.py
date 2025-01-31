#!/usr/bin/python
# author: Charlotte Bunne

# imports
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import torch
import time
import os
import pandas as pd
import seaborn as sns
from pylab import array
import torch.nn.functional as F

# internal imports
from model.utils import *
from model.data import *
from model.model_mlp import Generator, Adversary
from model.model_mlp import weights_init_adversary, weights_init_generator
from model.loss import gwnorm_distance
from model.loss import loss_procrustes
from model.sgw_pytorch_original import sgw_gpu_original
from model.risgw_original import risgw_gpu_original
from model.rarisgw import rarisgw_gpu
from model.rasgw_pytorch import rasgw_gpu
from model.iwrasgw_pytorch import iwrasgw_gpu
from baselines.ebsgw_trial import EBSGW
from baselines.rpsgw_trial import RPSGW
from baselines.dsgw_trial import DSGW
from baselines.ebrpsgw_trial import ebrpsgw_gpu
from baselines.maxsgw_trial import MaxSGW
# get arguments
FUNCTION_MAP = {'4mode': gaussians_4mode,
                '5mode': gaussians_5mode,
                '8mode': gaussians_8mode,
                '3d_4mode': gaussians_3d_4mode
                }
args = get_args()

# plotting preferences
matplotlib.rcParams['font.sans-serif'] = 'Times New Roman'
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.size'] = 10

# system preferences
torch.set_default_dtype(torch.double)
seed = np.random.randint(100)
np.random.seed(seed)
torch.manual_seed(seed)

# settings
batch_size = 256
z_dim = 256
lr = 0.0002
plot_every = 1000
niter = 10
epsilon = 0.01
ngen = 10
if args.advsy:
    lam = {'4mode': 0.001, '5mode': 0.0001}
else:
    lam = 0.01
beta = 1
stop_adversary = args.num_iter
l1_reg = args.l1reg
learn_c = args.advsy
train_iter = args.num_iter
modes = args.modes

if l1_reg:
    model = 'gwgan_gaussian_l1_{}_adversary_{}_{}'\
     .format(modes, learn_c, args.id)
else:
    model = 'gwgan_gaussian_{}_adversary_{}_{}'\
     .format(modes, learn_c, args.id)

simulation = FUNCTION_MAP[modes]

# data simulation
data, y = simulation(40000)
data_size = len(data)
data = np.concatenate((data, data[:batch_size, :]), axis=0)
y = np.concatenate((y, y[:batch_size]), axis=0)

save_fig_path = '2dto3d_out_sgw500_f_' + model
if not os.path.exists(save_fig_path):
    os.makedirs(save_fig_path)

real = data[:1000]
real_y = y[:1000]

fig1 = plt.figure(figsize=(2.4, 2))
if modes == '3d_4mode':
    df = pd.DataFrame({'x1': real[:, 0],
                       'x2': real[:, 1],
                       'x3': real[:, 2],
                       'in': real_y})
    ax1 = fig1.add_subplot(111, projection='3d')
    ax1.scatter3D(df.x1, df.x2, df.x3, c='#1B263B')
    ax1.set_zlim([-4, 4])
    view_1 = (25, -135)
    view_2 = (25, -45)
    init_view = view_2
    ax1.view_init(*init_view)
    ax1.set_zlabel('x3')
else:
    df = pd.DataFrame({'x1': real[:, 0],
                       'x2': real[:, 1],
                       'in': real_y})
    ax1 = fig1.add_subplot(111)
    sns.kdeplot(x=df.x1, y=df.x2, shade=True, cmap='Blues', n_levels=20, legend=False)
ax1.set_xlim([-4, 4])
ax1.set_ylim([-4, 4])
ax1.set_title(r'target')
fig1.tight_layout()
fig1.savefig(save_fig_path + '/real.pdf')

# Define whether you want 2D or 3D generation (for other modes)
generate_3d = True  # Set to False if you want 2D generation for other modes

# Define networks and parameters
if modes == '3d_4mode':
    generator = Generator(z_dim, output_dim=3)  # Generate 3D for '3d_4mode'
else:
    # Set generator to produce 3D or 2D based on generate_3d flag
    generator = Generator(z_dim, output_dim=3 if generate_3d else 2)

adversary = Adversary()
#generator=generator.to('cuda')
#adversary=adversary.to('cuda')

# weight initialisation
generator.apply(weights_init_generator)
adversary.apply(weights_init_adversary)

# create optimiser
g_optimizer = torch.optim.Adam(generator.parameters(), 5*lr)
c_optimizer = torch.optim.Adam(adversary.parameters(), lr)

# zero gradients
reset_grad(generator, adversary)

# sample for plotting
z_ex = sample_z(1000, z_dim)

# set iterator for plot numbering
i = 0

# learn with and without adversary
if learn_c:
    only_g = 0
else:
    only_g = train_iter

loss_history = list()
loss_orth = list()
loss_og = 0

reconstruction_losses_last_100 = []
iteration_times = []
start_time = time.time()
for it in range(train_iter):
    #start_time = time.time()
    train_c = ((it + 1) % (ngen + 1) == 0)

    start_idx = it * batch_size % data_size
    X_mb = data[start_idx:start_idx + batch_size, :]
    y_mb = y[start_idx:start_idx + batch_size]

    # sample points from latent space
    z = sample_z(batch_size, z_dim)

    # get data mini batch
    x = torch.from_numpy(X_mb[:batch_size, :])
    y_s = y_mb[:batch_size]

    if it <= only_g:
        for q in generator.parameters():
            q.requires_grad = True
        for p in adversary.parameters():
            p.requires_grad = False

        g = generator.forward(z)
        f_g = g
        f_x = x
    else:
        if train_c and it < stop_adversary:
            for q in generator.parameters():
                q.requires_grad = False
            for p in adversary.parameters():
                p.requires_grad = True

        else:
            for q in generator.parameters():
                q.requires_grad = True
            for p in adversary.parameters():
                p.requires_grad = False

        # result generator
        g = generator.forward(z)

        # result adversary
        f_x = adversary.forward(x,data=False)  #for 3d from 2d data=False
        f_g = adversary.forward(g,data=True)   #for 3d from 2d, data =True, else the data tag will not be present

    
    # compute inner distances
    D_g = get_inner_distances(f_g, metric='euclidean', concat=False)
    D_x = get_inner_distances(f_x, metric='euclidean', concat=False)

    # distance matrix normalisation
    D_x_norm = normalise_matrices(D_x)
    D_g_norm = normalise_matrices(D_g)

    if it==100 or it==500 or it==1000 or it==5000 or it==9999:  # Only store the last 100 iterations
        reconstruction_loss = gwnorm_distance((D_x, D_x_norm), (D_g, D_g_norm),epsilon, niter, loss_fun='square_loss', coupling=False)
        #reconstruction_loss = F.mse_loss(x, g)    #for 2d to 2d data 
        #reconstruction_losses_last_100.append(reconstruction_loss.item())
        print(f"Iteration [{it+1}/{train_iter}],  Loss GW: {reconstruction_loss.item():.4f}", "time now:",time.time()-start_time)
        #print(reconstruction_loss)
    # compute normalized gromov-wasserstein distance
    #loss_gw, T = gwnorm_distance((D_x, D_x_norm), (D_g, D_g_norm),epsilon, niter, loss_fun='square_loss', coupling=True)
    #loss_gw = gwnorm_distance((D_x, D_x_norm), (D_g, D_g_norm),epsilon, niter, loss_fun='square_loss', coupling=False)
    loss_gw = sgw_gpu_original(f_x.to('cuda'), f_g.to('cuda') ,'cuda',nproj=500,tolog=False,P=None)
    #loss_gw = rasgw_gpu(f_x.to('cuda'), f_g.to('cuda') ,'cuda',nproj=500,tolog=False,P=None)
    #loss_gw = rarisgw_gpu(f_x.to('cuda'), f_g.to('cuda'),'cuda' ,nproj=500,P=None,lr=0.001, max_iter=20, verbose=False, step_verbose=10, tolog=False, retain_graph=True)
    #loss_gw = risgw_gpu_original(f_x.to('cuda'), f_g.to('cuda') ,'cuda' ,nproj=500,P=None,lr=0.001, max_iter=10, verbose=False, step_verbose=10, tolog=False, retain_graph=True)
    #loss_gw = DSGW(f_x.to('cuda'), f_g.to('cuda') , L=5, kappa=10, p=2, s_lr=0.1, n_lr=10, device='cuda', nproj=500)
    #loss_gw = RPSGW(f_x.to('cuda'), f_g.to('cuda') , L=10  , nproj=500, p=2, kappa=50 )
    #loss_gw = MaxSGW(f_x.to('cuda'), f_g.to('cuda') , p=2, s_lr=0.1, n_lr=10, device='cuda', adam=False, nproj=500)
    #loss_gw = ebrpsgw_gpu(f_x.to('cuda'), f_g.to('cuda'),  L=10  , nproj=500, p=2, kappa=50 )
    #loss_gw = EBSGW(f_x.to('cuda'), f_g.to('cuda') , L=10, p=2, device='cuda')
    #loss_gw = rasgw_gpu(f_x.to('cuda'), f_g.to('cuda') , 'cuda', nproj=500,tolog=False,P=None)
    #loss_gw = iwrasgw_gpu(f_x.to('cuda'), f_g.to('cuda') , 'cuda', nproj=500,tolog=False,P=None)
    #loss_gw = rarisgw_gpu(f_x.to('cuda'), f_g.to('cuda') ,'cuda' ,nproj=500,P=None,lr=0.001, max_iter=20, verbose=False, step_verbose=10, tolog=False, retain_graph=True)
    if it < only_g:
        # train generator
        if l1_reg:
            loss_gen = loss_gw + lam * (torch.norm(g, p=1) - 2)
        else:
            loss_gen = loss_gw

        g_optimizer.zero_grad()
        loss_gen.backward()
        g_optimizer.step()
    else:
        # train adversary
        if it <= stop_adversary:
            adv_loss = loss_gw + lam[modes] * torch.norm(f_x - f_g, p=1)  # ADD adversary to generator term
            c_optimizer.zero_grad()
            adv_loss.backward()
            c_optimizer.step()

    if it==100 or it==500 or it==1000 or it==5000 or it==9999:# it % plot_every == 0 and it > 0:
        
        fig1 = plt.figure(figsize=(2.4, 2))
        if g.shape[1] == 3 :   # g.shape[1] == 3
            df = pd.DataFrame({'x1': g.detach()[:, 0],
                               'x2': g.detach()[:, 1],
                               'x3': g.detach()[:, 2],
                               })  # 'in': real_y was here
            ax1 = fig1.add_subplot(111, projection='3d')
            ax1.scatter3D(df.x1, df.x2, df.x3, c='#1B263B')
            ax1.set_zlim([-4, 4])
            ax1.set_xlabel('x1')
            ax1.set_ylabel('x2')
            ax1.set_zlabel('x3')
            ax1.set_title(f'generated: {it}/{train_iter}')
        else:
            df = pd.DataFrame({'x1': g.detach()[:, 0],
                               'x2': g.detach()[:, 1],
                               'in': real_y})
            ax1 = fig1.add_subplot(111)
            sns.kdeplot(x=df.x1, y=df.x2, shade=True, cmap='Blues', n_levels=20, legend=False)
            ax1.set_xlim([-4, 4])
            ax1.set_ylim([-4, 4])
            ax1.set_title(f'generated: {it}/{train_iter}')
        fig1.tight_layout()
        fig1.savefig(save_fig_path + f'/generated_{it}.pdf')
        #print(f"Iteration [{it+1}/{train_iter}],  Loss GW: {loss_gw.item():.4f}", "time now:",time.time()-start_time)
    # print progress
iteration_time = time.time() - start_time
iteration_times.append(iteration_time)

    #print(f"Iteration [{it+1}/{train_iter}], Time: {iteration_time:.4f}s, Loss GW: {loss_gw.item():.4f}")


# After the training loop, compute the average time per iteration
average_time_per_iteration = np.mean(iteration_times)

# Compute the variance of the time per iteration
# variance_time_per_iteration = np.var(iteration_times)

# Compute the standard deviation of the time per iteration (if you need it)
std_dev_time_per_iteration = np.std(iteration_times)

# Print the results
print(f"\nAverage time per iteration: {average_time_per_iteration:.4f} seconds.")
# print(f"Variance in time per iteration: {variance_time_per_iteration:.4f} seconds².")
print(f"Standard deviation in time per iteration: {std_dev_time_per_iteration:.4f} seconds.")

if 0:
    mean_loss = np.mean(reconstruction_losses_last_100)
    std_dev_loss = np.std(reconstruction_losses_last_100)

    print(f"Mean Reconstruction Loss (Last 100 iterations): {mean_loss}")
    print(f"Standard Deviation of Reconstruction Loss (Last 100 iterations): {std_dev_loss}")

# plot loss history
loss_history = loss_history
fig4 = plt.figure(figsize=(2.4, 2))
ax4 = fig4.add_subplot(111)
ax4.plot(np.arange(len(loss_history)) * 100, loss_history, 'k.')
ax4.set_xlabel('Iterations')
ax4.set_ylabel(r'$\overline{GW}_\epsilon$ loss')
plt.tight_layout()
fig4.savefig(save_fig_path + '/loss_history.pdf')

# plot orthogonal loss history
fig5 = plt.figure(figsize=(2.4, 2))
ax5 = fig5.add_subplot(111)
ax5.plot(np.arange(len(loss_orth)) * 100, loss_orth, 'k.')
ax5.set_xlabel('Iterations')
ax5.set_ylabel(r'$R_\beta(f_\omega(X), X)$ loss')
plt.tight_layout()
fig5.savefig(save_fig_path + '/loss_orth.pdf')

# save losses and other info after training
np.save(save_fig_path + '/losses.npy', np.array(reconstruction_losses_last_100))
np.save(save_fig_path + '/iteration_times.npy', np.array(iteration_times))

