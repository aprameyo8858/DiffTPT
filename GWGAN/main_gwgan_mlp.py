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
plot_every = 10000
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

save_fig_path = 'correct_out_iwrpsgw_f_' + model
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

# define networks and parameters
generator = Generator()
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
gw_loss_100 = []
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
        f_x = adversary.forward(x,data=True)
        f_g = adversary.forward(g,data=False)

    
    # compute inner distances
    D_g = get_inner_distances(f_g, metric='euclidean', concat=False)
    D_x = get_inner_distances(f_x, metric='euclidean', concat=False)

    # distance matrix normalisation
    D_x_norm = normalise_matrices(D_x)
    D_g_norm = normalise_matrices(D_g)

    if it==100 or it==500 or it==1000 or it==5000 or it==9999: #it >= train_iter - 100 # Only store the last 100 iterations
        D_g = get_inner_distances(f_g, metric='euclidean', concat=False)
        D_x = get_inner_distances(f_x, metric='euclidean', concat=False)

        # distance matrix normalisation
        D_x_norm = normalise_matrices(D_x)
        D_g_norm = normalise_matrices(D_g)

        #time_now=end_time-start_time
        gw_loss = gwnorm_distance((D_x, D_x_norm), (D_g, D_g_norm),epsilon, niter, loss_fun='square_loss', coupling=False)
        end_time = time.time()
        time_now=end_time-start_time
        #reconstruction_loss = F.mse_loss(x, g)    #for 2d to 2d data 
        #gw_loss_100.append(gw_loss.item())
        #reconstruction_losses_last_100.append(reconstruction_loss.item())
        print("iteration number,",it," gw+loss:" , gw_loss, "time :",time_now)
    # compute normalized gromov-wasserstein distance
    #loss_gw, T = gwnorm_distance((D_x, D_x_norm), (D_g, D_g_norm),epsilon, niter, loss_fun='square_loss', coupling=True)
    #loss_gw = gwnorm_distance((D_x, D_x_norm), (D_g, D_g_norm),epsilon, niter, loss_fun='square_loss', coupling=False)
    #loss_gw = sgw_gpu_original(f_x.to('cuda'), f_g.to('cuda') ,'cuda',nproj=500,tolog=False,P=None)
    #loss_gw = rasgw_gpu(f_x.to('cuda'), f_g.to('cuda') ,'cuda',nproj=500,tolog=False,P=None)
    #loss_gw = rarisgw_gpu(f_x.to('cuda'), f_g.to('cuda'),'cuda' ,nproj=500,P=None,lr=0.001, max_iter=20, verbose=False, step_verbose=10, tolog=False, retain_graph=True)
    #loss_gw = risgw_gpu_original(f_x.to('cuda'), f_g.to('cuda') ,'cuda' ,nproj=500,P=None,lr=0.001, max_iter=10, verbose=False, step_verbose=10, tolog=False, retain_graph=True)
    #loss_gw = DSGW(f_x.to('cuda'), f_g.to('cuda') , L=5, kappa=10, p=2, s_lr=0.1, n_lr=10, device='cuda', nproj=500)
   # loss_gw = RPSGW(f_x.to('cuda'), f_g.to('cuda') , L=10  , nproj=500, p=2, kappa=50 )
    #loss_gw = MaxSGW(f_x.to('cuda'), f_g.to('cuda') , p=2, s_lr=0.1, n_lr=10, device='cuda', adam=False, nproj=500)
    loss_gw = ebrpsgw_gpu(f_x.to('cuda'), f_g.to('cuda'),  L=10  , nproj=500, p=2, kappa=50 )
    #loss_gw = EBSGW(f_x.to('cuda'), f_g.to('cuda') , L=10, p=2, device='cuda')
    #loss_gw = rasgw_gpu(f_x.to('cuda'), f_g.to('cuda') , 'cuda', nproj=500,tolog=False,P=None)
    #loss_gw = iwrasgw_gpu(D_x.to('cuda'), D_g.to('cuda') , 'cuda', nproj=500,tolog=False,P=None)
    #loss_gw = iwrasgw_gpu(f_x.to('cuda'), f_g.to('cuda') , 'cuda', nproj=5000,tolog=False,P=None)
    #loss_gw = rarisgw_gpu(f_x.to('cuda'), f_g.to('cuda') ,'cuda' ,nproj=500,P=None,lr=0.001, max_iter=20, verbose=False, step_verbose=10, tolog=False, retain_graph=True)
    if it < only_g:
        # train generator
        if l1_reg:
            loss_gen = loss_gw + lam * (torch.norm(g, p=1) - 2)
        else:
            loss_gen = loss_gw
        loss_gen.backward()

        # parameter updates
        g_optimizer.step()

        # zero gradients
        g_optimizer.zero_grad()

    else:
        if train_c and it < stop_adversary:
            loss_og = loss_procrustes(f_x, x, cuda=False)
            loss_adv = -loss_gw + beta * loss_og
            # train adversary
            loss_adv.backward()

            # parameter updates
            c_optimizer.step()
            # zero gradients
            reset_grad(generator, adversary)

        else:
            # train generator
            if l1_reg:
                loss_gen = loss_gw + lam[modes] * (torch.norm(g, p=1) - 2)
            else:
                loss_gen = loss_gw
            loss_gen.backward()

            # parameter updates
            g_optimizer.step()
            # zero gradients
            reset_grad(generator, adversary)

    # plotting
    if it==100 or it==500 or it==1000 or it==5000 or it==9999:#(it+1) % plot_every == 0
        # get generator example
        g_ex = generator.forward(z_ex)
        if it >= only_g:
            f_gx = adversary.forward(g_ex)
            f_gx = f_gx.detach().numpy()
            f_dx = adversary.forward(torch.from_numpy(real),data=True)
            f_dx = f_dx.detach().numpy()
        g_ex = g_ex.detach().numpy()

        # plotting
        fig2 = plt.figure(figsize=(2.4, 2))
        ax2 = fig2.add_subplot(111)
        result = pd.DataFrame({'x1': g_ex[:, 0],
                               'x2': g_ex[:, 1]})
        #for now 
        sns.kdeplot(x=result.x1, y=result.x2,
                    fill=True, cmap='Blues', n_levels=20, legend=False)
        # ax2.set_title(r'$g_\theta(Z)$')
        ax2.set_title(r'iteration {}'.format((it+1)))
        plt.tight_layout()
        fig2.savefig(os.path.join(save_fig_path, 'gen_{}.pdf'.format(
                     str(i).zfill(3))))

        if it >= only_g:
            fig6 = plt.figure(figsize=(4.5, 2))
            features = pd.DataFrame({'g1': f_gx[:, 0],
                                     'g2': f_gx[:, 1],
                                     'd1': f_dx[:, 0],
                                     'd2': f_dx[:, 1]
                                     })
            ax1 = fig6.add_subplot(121)
            sns.kdeplot(x=features.g1, y=features.g2,
                        fill=True, cmap='Greys', n_levels=20, legend=False)
            # ax1.set_title(r'$f_\omega(g_\theta(Z))$')
            ax1.set_xlim([-4, 4])
            ax1.set_ylim([-4, 4])
            ax1.set_title(r' ')

            ax2 = fig6.add_subplot(122)
            sns.kdeplot(x=features.d1, y=features.d2,
                        fill=True, cmap='Greys', n_levels=20, legend=False)
            ax2.set_xlim([-4, 4])
            ax2.set_ylim([-4, 4])
            ax2.set_title(r' ')
            plt.tight_layout(pad=1)
            fig6.savefig(os.path.join(save_fig_path, 'feature_{}.pdf'.format(
                         str(i).zfill(3))))

        #fig3, ax = plt.subplots(1, 3, figsize=(6.9, 2))
        #ax0 = ax[0].imshow(T.detach().numpy(), cmap='RdBu_r')
        #colorbar(ax0)
        #ax1 = ax[1].imshow(D_g.detach().numpy(), cmap='Blues')
        #colorbar(ax1)
        #ax2 = ax[2].imshow(D_x.detach().numpy(), cmap='Blues')
        #colorbar(ax2)
        #ax[0].set_title(r'$T$')
        #ax[1].set_title(r'Pairwise Distances of $f_\omega(g_\theta(Z))$')
        #ax[2].set_title(r'Pairwise Distances of $f_\omega(X)$')
        #plt.tight_layout(pad=1)
        #fig3.savefig(os.path.join(save_fig_path, 'ccc_{}.pdf'.format(
                #str(i).zfill(3))), bbox_inches='tight')

        plt.close('all')
        print('iter:', it+1, 'GW loss:', loss_gw, 'Orth. loss', loss_og)
        i += 1

        loss_history.append(loss_gw)
        loss_orth.append(loss_og)
        # After the iteration is finished, record the end time
end_time = time.time()

        # Calculate the time taken for the iteration
iteration_time = end_time - start_time

        # Store the iteration time in the list
iteration_times.append(iteration_time)


# After the training loop, compute the average time per iteration
average_time_per_iteration = np.mean(iteration_times)

# Compute the variance of the time per iteration
#variance_time_per_iteration = np.var(iteration_times)

# Compute the standard deviation of the time per iteration (if you need it)
std_dev_time_per_iteration = np.std(iteration_times)

# Print the results
print(f"\nAverage time per iteration: {average_time_per_iteration:.4f} seconds.")
#print(f"Variance in time per iteration: {variance_time_per_iteration:.4f} seconds².")
print(f"Standard deviation in time per iteration: {std_dev_time_per_iteration:.4f} seconds.")
if len(reconstruction_losses_last_100) > 0:
    mean_loss = np.mean(reconstruction_losses_last_100)
    std_dev_loss = np.std(reconstruction_losses_last_100)
    gw_mean = np.mean(gw_loss_100)
    gw_std = np.std( gw_loss_100)
    print(f"Mean Reconstruction Loss (Last 100 iterations): {mean_loss}")
    print(f"Standard Deviation of Reconstruction Loss (Last 100 iterations): {std_dev_loss}")
    print(f"Mean GW Loss (Last 100 iterations): {gw_mean}")
    print(f"Standard Deviation of GW Loss (Last 100 iterations): {gw_std}")
# plot loss history
#loss_history = loss_history
# Ensure loss_history is properly handled, even if it's a list or tensor
if isinstance(loss_history, torch.Tensor):
    if loss_history.requires_grad:
        loss_history = loss_history.detach().cpu().numpy()  # Detach from the graph, move to CPU, and convert to NumPy
    else:
        loss_history = loss_history.cpu().numpy()  # Move to CPU and convert to NumPy array
elif isinstance(loss_history, list) or isinstance(loss_history, np.ndarray):
    loss_history = np.array(loss_history)  # In case it's a list or NumPy array, ensure it's a NumPy array

# Ensure loss_orth is properly handled, even if it's a list or tensor
if isinstance(loss_orth, torch.Tensor):
    if loss_orth.requires_grad:
        loss_orth = loss_orth.detach().cpu().numpy()  # Detach from the graph, move to CPU, and convert to NumPy
    else:
        loss_orth = loss_orth.cpu().numpy()  # Move to CPU and convert to NumPy array
elif isinstance(loss_orth, list) or isinstance(loss_orth, np.ndarray):
    loss_orth = np.array(loss_orth)  # In case it's a list or NumPy array, ensure it's a NumPy array

fig4 = plt.figure(figsize=(2.4, 2))
ax4 = fig4.add_subplot(111)
ax4.plot(np.arange(len(loss_history))*100, loss_history, 'k.')
ax4.set_xlabel('Iterations')
ax4.set_ylabel(r'$\overline{GW}_\epsilon$ loss')
plt.tight_layout()
fig4.savefig(save_fig_path + '/loss_history.pdf')


fig5 = plt.figure(figsize=(2.4, 2))
ax5 = fig5.add_subplot(111)
ax5.plot(np.arange(len(loss_orth))*100, loss_orth, 'k.')
ax5.set_xlabel('Iterations')
ax5.set_ylabel(r'$R_\beta(f_\omega(X), X)$ loss')
plt.tight_layout()
fig5.savefig(save_fig_path + '/loss_orth.pdf')
