import torch
import torch.nn as nn
import numpy as np
import sys
sys.path.append('./GeometricalDisentangling')
from generate_data import get_data
import geoopt
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

def orthogonalize(w):
    # make it a matrix with det = 1
    u, s, v = torch.svd(w)
    return torch.mm(u, v.t())

def torch_trapezoidal_integral(f, a, b, n = 2000):
    '''
    Compute the integral of a function f
    using the trapezoidal rule.
    '''
    x = torch.linspace(a, b, n)
    y = f(x)
    return (y[:, 0] + y[:, -1] + 2 * y[:, 1:-1].sum()) * (b - a) / (2 * n)

def log_Bassel_function(k):
    '''
    Modiefied Bessel function I_0,
    performs numerical integration.
    '''
    integrand = lambda x: torch.exp(-k + k * torch.cos(x)) # rewriting the integral
    # taking exp(k) out of the integral, then taking the log of it!
    
    # This should work because this intrgeral is kind of trivial (unidimensional)

    norm_constant = torch.clamp(torch_trapezoidal_integral(integrand, 0, torch.pi).unsqueeze(1), min = 1e-8)
    log_norm_constant = torch.log(norm_constant / torch.pi) + k
    return log_norm_constant

def generate_omega_vectors(s_len= 3, J = 50):
    # Generate a random matrix with entries in [1, max_value]
    matrix = torch.randint(1, J + 1, (s_len, J //2))
    return matrix

def T(s, omega_matrix):
    '''
    Mapping from s to the vector of cos and sin
    as a linear combination of the integer basis functions.
    '''
    angles = s @ omega_matrix # note that s \in [0, 2\pi]^s_len
    # generate first vector cos(angles)
    cos_angles = torch.cos(angles)
    sin_angles = torch.sin(angles)
    # put in order, this is, by cos sin pairs 
    # (cos(\theta_1), sin(\theta_1), cos(\theta_2), sin(\theta_2), ...)
    interleaved = torch.empty((sin_angles.shape[0], 2 * sin_angles.shape[1]), dtype = s.dtype)
    interleaved[:, 0::2] = cos_angles
    interleaved[:, 1::2] = sin_angles
    return interleaved

def log_coupled_norm_constant(eta_hat_vector, omega_matrix, s_len = 2):
    '''
    Implies computing an integral over s. Potentially
    having multiple dimensions.

    I am going to do simpler integration, so just computing cubes on a grid
    i.e. Riemanian sum.
    '''
    
    integrand = lambda s: torch.exp(
        torch.clamp(torch.matmul(eta_hat_vector, T(s, omega_matrix).t()), 0, 80)
        ) # clamp to avoid inf (very bad solution)
    assert s_len == 2, 'Not implemented for s_len != 2'
    # make a grid of s 
    s_1 = torch.linspace(0, 2 * torch.pi, 100)
    s_2 = torch.linspace(0, 2 * torch.pi, 100)
    s = torch.stack(torch.meshgrid(s_1, s_2,  indexing='ij'), dim = 2).reshape(-1, 2)
    integrand_evaluations = integrand(s)
    # check if any inf
    if torch.any(torch.isinf(integrand_evaluations)):
        print('Warning: inf values in integrand')
        import pdb; pdb.set_trace()
    # compute the integral using riemanian sum
    integral = torch.mean(integrand_evaluations, dim = 1) * (2 * torch.pi) ** 2
    return torch.log(integral)

def compute_posterior_eta(u, v, sigma = 1, prior_k = 0):
    # Returns a 2d vector! The coordinates on the unit circle
    # of the angle between u and v. Scaled by part_1
    assert prior_k == 0, 'Not implemented with prior_k != 0'

    #part_1 = (torch.norm(u, dim = 1)*torch.norm(v, dim = 1) / sigma ** 2).unsqueeze(1)
    #dot_prod = torch.sum(u * v, dim = 1)
    #angle = torch.acos(torch.clamp(dot_prod / (torch.norm(u, dim = 1) * torch.norm(v, dim = 1)), -0.9999, 0.9999))
    #part_2 = torch.stack((torch.cos(angle), torch.sin(angle)), dim = 1)
    #return part_1 * part_2
    u1, u2 = u[:, 0], u[:, 1]
    v1, v2 = v[:, 0], v[:, 1]
    
    eta_hat_1 = (1/sigma ** 2) * (u1 * v1 + u2 * v2)
    eta_hat_2 = (1/sigma ** 2) * (u1 * v2 - u2 * v1)

    return torch.stack((eta_hat_1, eta_hat_2), dim = 1)

def get_w_subspace(w, subspace_i):
    '''
    Get the i-th subspace of the orthogonal matrix
    '''
    D = w.shape[-1]
    assert D % 2 == 0, 'D must be even!'

    W_j = w[:, 2 * subspace_i:2 * subspace_i + 2]
    # Which is the two columns that get a linear combination of the image into a subspace
    # which is then rotated.

    return W_j

def log_proba_y_given_x(y, x, w, omega_matrix = None, sigma = 0.1, prior_k = 0, coupled = False, s_len = 2):
    '''
    Compute the probability of y given x. Essentially,
    the - loss function.

    This is the "decoupled" model, which means
    that there are D/2 parameters. Assuming D
    is even!

    A "maximal torus" has J = D/2 degrees of
    freedom.
    
    (TODO, show the steps to compute this integral!)
    Parameters
    ----------
    y : torch.Tensor
        The target tensor image
    x : torch.Tensor
        The input tensor image
    W : torch.Tensor
        The orthogonal matrix W
    eta : torch.Tensor
        The prior angle
    mu_hat : torch.Tensor
        The posterior angle
    sigma : float
        The standard deviation of the noise. 
        It is not estimated! (Why?)
    '''
    assert prior_k == 0, 'Not implemented with prior_k != 0'
    prior_k = torch.tensor(0)
    D = w.shape[-1]
    assert D % 2 == 0, 'D must be even!'

    #norms_sum = torch.norm((w.t() @ x.t()).t(), dim = 1) ** 2 + torch.norm(y, dim = 1) ** 2
    norms_sum = torch.norm(x, dim = 1) ** 2 + torch.norm(y, dim = 1) ** 2
    scaled_norms = - 1/(2 * sigma ** 2) * norms_sum
    scaled_norms -= torch.tensor(D) * torch.log(torch.tensor(2 * torch.pi) * sigma ** 2).unsqueeze(0)
    

    # now the angle, the ratio of normalization
    # constants (evidence) for the prior and the posterior
    
    # First, I need to calculate the posterior for \eta. This is,
    # the updated average angle.

    # the complicated part in this segment is that this is calculated 
    # over the irreducible parts, so for independent subspace.
    if not coupled:
        quotients = []  
    else:
        etas_hat = []
    for subspace_i in range(D // 2):
        W_j = get_w_subspace(w, subspace_i)
        u_j = (W_j.t() @ x.t()).t() # To handle the batch
        v_j = (W_j.t() @ y.t()).t() # To handle the batch
        eta_hat =  compute_posterior_eta(u_j, v_j, sigma = sigma, prior_k = prior_k)        
        # compute normlization constants
        #norm_prior = Bassel_function(prior_k) Note: this is not needed
        #norm_prior = 1 # BECAUSE THIS IS FIXED AT 0!! (uniform prior) The integral will give 0
        if not coupled:
            log_norm_posterior = log_Bassel_function(torch.norm(eta_hat, dim = 1, keepdim= True))
            quotients.append(log_norm_posterior) #- torch.log(norm_prior)) # Cuz log(1) = 0
        else:
            etas_hat.append(eta_hat)


    #print("Quotients:")
    #print(quotients)
    #import pdb; pdb.set_trace()
    if not coupled:
        anlge_part = torch.sum(torch.stack(quotients).squeeze(2).t(), dim = 1)
    else:
        # take multi-dimensional integral over s
        # concatenate such that I have [cos(\theta), sin(\theta), ...]
        eta_hat_vector = torch.cat(etas_hat, dim = 1)
        # this comes as B x D //2 x 2. That is interesting because the first dimension is cos(\theta)
        # the second is sin(\theta).
        anlge_part = log_coupled_norm_constant(eta_hat_vector, omega_matrix, s_len = s_len)

    log_lik = scaled_norms + anlge_part
    
    return log_lik

def log_loss(y, x, w, omega_matrix, sigma = 0.1, prior_k = 0, coupled = False):
    '''
    Compute the log loss function.
    '''
    return - torch.mean(log_proba_y_given_x(y, x, w, omega_matrix,  sigma = sigma, prior_k = prior_k, coupled = False))

def compute_probas(y, x, w, sigma = 0.1, prior_k = 0):
    return torch.exp(log_proba_y_given_x(y, x, w, sigma = sigma, prior_k = prior_k))

def train_model(train, references, w, omega_matrix, lr = 0.5, n_epochs = 1, batch_size = 256):
    w_optimizer = geoopt.optim.RiemannianAdam([w], lr=lr)

    # generate data loader with train and reference
    train_loader = \
        torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(train, references),
              batch_size=batch_size, shuffle=True)
    losses = []
    for epoch in range(n_epochs):
        loss_list = []
        for x, y in train_loader:
            w_optimizer.zero_grad()
            loss = log_loss(y, x, w, omega_matrix, sigma=0.1, prior_k=0, coupled = False)
            loss.backward()
            w_optimizer.step()
            loss_list.append(loss.item())
        losses.append(np.mean(loss_list))
        # assert that w is orthogonal
        if epoch % 10 == 0:
            print(torch.allclose(w.t() @ w, torch.eye(w.shape[-1]), 1e-3, 1e-3))#, 'W is not orthogonal!'
            print(f'Epoch {epoch}, Loss: {loss.item()}')
        plt.plot(losses)
        plt.savefig('loss.png')
    return w

def construct_operator(w, x, y):
    '''
    Construct the operator that maps x to y
    '''
    D = w.shape[-1]
    assert D % 2 == 0, 'D must be even!'

    operator = torch.zeros(w.shape[1], w.shape[1])
    for subspace_i in range(D // 2):
        W_j = get_w_subspace(w, subspace_i)
        u_j = (W_j.t() @ x.t()).t() # To handle the batch
        v_j = (W_j.t() @ y.t()).t() # To handle the batch
        eta_hat = compute_posterior_eta(u_j, v_j, sigma = 0.1, prior_k = 0)     
        # eta hat is cos, sin of the angle
        # construct the rotation matrix
        rotation_2d_block = torch.zeros(2, 2)
        # rotation matrix is:
        # cos(theta) -sin(theta)
        # sin(theta) cos(theta)
        # get polar form of eta
        theta = torch.atan2(eta_hat[:, 1], eta_hat[:, 0])
        magnitude = torch.norm(eta_hat, dim = 1)
        rotation_2d_block[0, 0] = torch.cos(theta)
        rotation_2d_block[0, 1] = -torch.sin(theta)
        rotation_2d_block[1, 0] = torch.sin(theta)
        rotation_2d_block[1, 1] = torch.cos(theta)
        # put in the block diagonal
        operator[2 * subspace_i:2 * subspace_i + 2, 2 * subspace_i:2 * subspace_i + 2] = rotation_2d_block #* magnitude.unsqueeze(1)
    return operator
   
def rotate(w, x, y):
    R = construct_operator(w, x, y)
    rotated = w @ R @ (w.t() @ x.t())
    return rotated

if __name__ == "__main__":
    # Get the data
    train_data, test_data, reference_train, reference_test = get_data(None)
    # initialize orthogonal w
    w = torch.randn(28*28, 100)
    w = orthogonalize(w)
    fig, axs = plt.subplots(10, 10, figsize=(20, 20))
    #for i in range(100):
    #    axs[i//10, i%10].imshow(w[:, i].detach().numpy().reshape(28, 28), cmap='gray')
    #plt.show()
    print(torch.allclose(w.t() @ w, torch.eye(w.shape[-1]), 1e-4, 1e-4))
    w_manifold = geoopt.manifolds.Stiefel()
    w = geoopt.ManifoldParameter(w, manifold=w_manifold,requires_grad=True)
    print(w.shape)

    # initialize omega matrix
    omega_matrix = generate_omega_vectors(s_len=2, J = 100).float()
    # Get one example
    x = train_data.float()
    y = reference_train.float()
    # select 20 pairs of random images
    idx = torch.randint(0, x.shape[0], (20,))
    # plot them in a grid, reference and train
    fig, axs = plt.subplots(20, 2, figsize=(10, 40))
    for i in range(20):
        # plot train
        axs[i, 0].imshow(x[idx[i]].reshape(28, 28), cmap='gray')
        axs[i, 0].set_title('Train')
        axs[i, 0].axis('off')
        # plot reference
        axs[i, 1].imshow(y[idx[i]].reshape(28, 28), cmap='gray')
        axs[i, 1].set_title('Reference')
        axs[i, 1].axis('off')
    plt.tight_layout()
    plt.savefig('train_reference.png')    
    plt.close("all")
    # convert to tensor
    #w = w.clone().detach().requires_grad_(True)
    w = train_model(x, y, w, omega_matrix)
    # plot w in a grid 1 plot for each column
    fig, axs = plt.subplots(10, 10, figsize=(20, 20))
    for i in range(100):
        axs[i//10, i%10].imshow(w[:, i].detach().numpy().reshape(28, 28), cmap='gray')
    plt.savefig('w.png')
    plt.close("all")
    # TODO plot a rotation ! 
    # plot reference, x and rotated for 5 different examples
    fig, axs = plt.subplots(5, 4, figsize=(10, 20))
    for i in range(5):
        rotated = rotate(w, x[idx[i:i+1]], y[idx[i:i+1]])
        axs[i, 0].matshow(x[idx[i]].reshape(28, 28))
        axs[i, 0].set_title('Train')
        axs[i, 0].axis('off')
        axs[i, 1].matshow(y[idx[i]].reshape(28, 28))
        axs[i, 1].set_title('Reference')
        axs[i, 1].axis('off')
        axs[i, 2].matshow(rotated.detach().numpy().reshape(28, 28))
        axs[i, 2].set_title('Rotated')
        axs[i, 2].axis('off')
        axs[i, 3].matshow((w @ w.t() @ x[idx[i:i+1]].t()).detach().numpy().reshape(28, 28))
        axs[i, 3].set_title('Reconstructed no rotation')
        axs[i, 3].axis('off')
    plt.tight_layout()
    plt.savefig('rotated.png')

    plt.matshow((w @ w.t() @ x[idx[0:1]].t()).detach().numpy().reshape(28, 28))
    plt.show()