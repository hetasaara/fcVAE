import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable


def simulate(N: int = 200, K: int = 6, seed: int = 12345):
    """Generate artificial data.

    :param N: number of cells per cluster
    :param K: number of clusters
    :param seed: random seed
    """
    np.random.seed(seed)
    z, labels = simulate_z(N, K)
    x = simulate_x(z)
    return x, z, labels

def simulate_z(N: int, K: int):
    """Generate latent representations.
    """
    R = 1
    theta = np.linspace(0, 2*np.pi, K+1)
    theta = theta[0:K] + 0.3*np.random.normal(size=K)
    m1 = R*np.cos(theta)
    m2 = R*np.sin(theta)
    m = np.vstack((m1, m2)).T
    M = np.tile(m, (N, 1))
    s = 0.2*np.exp(-0.7*np.random.normal(size=(K, 2)))
    S = np.tile(s, (N, 1))
    z = np.random.normal(loc=M, scale=S)
    lab = np.linspace(0, K-1, K)
    labels = np.tile(lab, N)
    return z, labels

def simulate_x(z):
    """Map latent representation z to measurements x
    """
    n_dim = 16
    h = create_h(z)
    w = 0.5*np.random.normal(size=(h.shape[1], n_dim))
    x = h @ w
    return x

def dist(z, a):
    """Compute euclidean distance of each row of z from point a"""
    d0 = z[:, 0] - a[0]
    d1 = z[:, 1] - a[1]
    d = np.sqrt(d0**2 + d1**2)
    return d

def create_h(z):
    """Create intermediate representation h from z"""
    r = 1
    a = [0,0]
    b = [r,r]
    c = [r,-r]
    d = [-r,r]
    e = [-r,-r]
    s = 1
    h0 = np.exp(-s*dist(z, a)**2)
    h1 = np.exp(-s*dist(z, b)**2)
    h2 = np.exp(-s*dist(z, c)**2)
    h3 = np.exp(-s*dist(z, d)**2)
    h4 = np.exp(-s*dist(z, e)**2)
    return np.vstack((h0,h1,h2,h3,h4)).T

def plot_latent(z, labels):
    """Visualize generated latent representations.
    """
    labs = np.unique(labels)
    plt.figure(figsize=(6,6))
    for lab in labs:
        indices = np.where(labels==lab)[0]
        plt.scatter(z[indices, 0], z[indices, 1])
    plt.show()

def plot_x(x, z):
    """Plot x using color.
    """
    fig, ax = plt.subplots(figsize=(10, 8), nrows=4, ncols=4)
    for i in range(0, 4):
        for j in range(0, 4):
            r = j + 4*i
            c = x[:, r]
            im = ax[i,j].scatter(z[:, 0], z[:, 1], c = c)
            ax[i,j].set_title('x[:, ' + str(r) + ']')
            ax[i,j].axis('off')
            divider = make_axes_locatable(ax[i,j])
            cax = divider.append_axes('right', size='5%', pad=0.05)
            fig.colorbar(im, cax=cax, orientation='vertical')
    plt.show()

