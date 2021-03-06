import numpy as np
import torch
from matplotlib import pyplot as plt
import seaborn as sns


def draw_plot(save_name, **kwargs):
    """Function to be used when a plot is to be shown or saved.
    """
    if save_name is None:
        plt.show()
    else:
        save_path = os.path.join(get_output_dir(), save_name)
        log_message('Saving figure to ' + save_path)
        plt.savefig(save_path, **kwargs)
        plt.close()


def pair_plot(z, colors, panel_size=3.8, marker_size=10, alpha=0.7, save_name=None, **kwargs):
    """Pair plot with all dimension pairs.

    :param z: a numpy array with shape [n_cells, n_latent_dim]
    :type z: numpy.ndarray
    :param panel_size: size of one panel
    :type panel_size: float
    :param marker_size: marker size in the scatter plots
    :type marker_size: int
    :param alpha: marker alpha
    :type alpha: float
    :param colors: cell colors in a numpy array with length n_cells
    :type colors: numpy.ndarray
    :param save_name: filename for saving figure:
    :type save_name: str
    """
    d = z.shape[1]
    nplots = int(d*(d-1)/2)
    nrows, ncols = determine_nrows_ncols(nplots)
    figsize = (panel_size * ncols, panel_size * nrows)
    fix, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
    counter = 0
    for i in range(0, d):
        for j in range(i+1, d):
            c = counter % ncols
            r = int(np.floor(counter/ncols))
            title = 'dim ' + str(i+1) + ' vs. dim ' + str(j+1)
            if nrows > 1:
                ax[r, c].scatter(z[:, i], z[:, j], c=colors, s=marker_size, alpha=alpha)
                ax[r, c].set_title(title)
            else:
                ax[c].scatter(z[:, i], z[:, j], c=colors, s=marker_size, alpha=alpha)
                ax[c].set_title(title)
            counter += 1
    draw_plot(save_name, **kwargs)


def determine_nrows_ncols(nplots: int):
    """Determine number of rows and columns a grid of subplots.

    :param nplots: total number of subplots
    :type nplots: int
    """
    if nplots < 2:
        raise ValueError("nplots should be at least 2!")
    if nplots < 4:
        ncols = nplots
    elif nplots < 5:
        ncols = 2
    elif nplots < 13:
        ncols = 3
    else:
        ncols = 4
    nrows = int(np.ceil(nplots / ncols))
    return nrows, ncols


def colors10(n_colors: int = 10):
    """
    Returns the n_colors first colors from the category10 colormap
    (https://github.com/vega/vega/wiki/Scales#scale-range-literals)

    Parameters
    ----------
    n_colors: int
        an integer determining the number of colors

    Returns
    ---------
    colors: list
        a list of length n_colors
    """
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
              '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    return colors[0:n_colors]


def colors20(n_colors: int = 20):
    """
    Returns the n_colors first colors from the category20 colormap
    (https://github.com/vega/vega/wiki/Scales#scale-range-literals)

    Parameters
    ----------
    n_colors: int
        an integer determining the number of colors

    Returns
    ---------
    colors: list
        a list of length n_colors
    """
    colors = ['#1f77b4', '#aec7e8', '#ff7f0e', '#ffbb78', '#2ca02c',
              '#98df8a', '#d62728', '#ff9896', '#9467bd', '#c5b0d5',
              '#8c564b', '#c49c94', '#e377c2', '#f7b6d2', '#7f7f7f',
              '#c7c7c7', '#bcbd22', '#dbdb8d', '#17becf', '#9edae5']
    return colors[0:n_colors]


def plot_trainer_history(trainer, i_start: int = 0, save_name=None):
    """
    Plot loss history of a Trainer
    """

    fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(17, 8))
    titles = ["Vae loss", "KL divergence", "Reconstruction loss", "Fooling loss", "Discriminator loss"]

    for k in range(0, 3):
        r = k
        loss1 = trainer.TRAIN_HISTORY[i_start:, r]
        loss2 = trainer.TEST_HISTORY[i_start:, r]
        x = np.linspace(0, trainer.n_epochs, trainer.n_epochs)[i_start:]
        ax[0, k].plot(x, loss1)
        ax[0, k].plot(x, loss2)
        ax[0, k].legend(('Train', 'Test'))
        ax[0, k].set_title(titles[r])

    for k in range(0, 2):
        r = 3 + k
        loss1 = trainer.TRAIN_HISTORY[i_start:, r]
        loss2 = trainer.TEST_HISTORY[i_start:, r]
        x = np.linspace(0, trainer.n_epochs, trainer.n_epochs)[i_start:]
        ax[1, k].plot(x, loss1)
        ax[1, k].plot(x, loss2)
        ax[1, k].set_xlabel('Epoch')
        ax[1, k].legend(('Train', 'Test'))
        ax[1, k].set_title(titles[r])

    ax[1,2].get_xaxis().set_visible(False)
    ax[1,2].get_yaxis().set_visible(False)

    if save_name is None:
        plt.show()
    else:
        print('Saving figure as ' + save_name)
        plt.savefig(save_name)
        plt.show()
        plt.close

def plot_probabilities(trainer, i_start: int = 0, save_name=None):
    '''
    Plot the discrminator probabilities during training.
    '''
    plt.figure(figsize=(8, 4))
    labels = []
    for i in range(trainer.n_tubes):
        prob1 = [item[i] for item in trainer.TRAIN_PROBS]      
        x = np.linspace(0, trainer.n_epochs, trainer.n_epochs)
        plt.plot(x, prob1)
        plt.xlabel('Epoch')
        plt.ylabel('Probability')
        labels.append(i+1)
    plt.title("Training probabilities")
    plt.legend(labels)
    
    if save_name is None:
        plt.show()
    else:
        print('Saving figure as ' + save_name)
        plt.savefig(save_name)
        plt.show()
        plt.close

def plot_joint_aml(trainer, latent2d, latents, save_name=None):    
    colors = sns.color_palette(n_colors=30) 
    
    plt.figure(figsize=(10, 10))
    for i in range(trainer.n_tubes):
        zs = latent2d[i*latents[i].shape[0] : (i+1)*latents[i].shape[0]]
        plt.scatter(zs[:,0], zs[:, 1], color=colors[i], label=i+1, alpha=0.5, s=1)
    plt.legend()
    
    if save_name is None:
        plt.show()
    else:
        print('Saving figure as ' + save_name)
        plt.savefig(save_name)
        plt.show()
        plt.close

def plot_shared_markers_aml(trainer, latent2d, latents, datasets, save_name=None):           
    titles = ["FSC", "SSC", "CD45"]
    for i in range(trainer.n_tubes):
        fig = plt.figure(figsize=(20,5))
        for j in range(1,4):
            zs = latent2d[i*latents[i].shape[0] : (i+1)*latents[i].shape[0]]
            plt.subplot(1,3,j)
            t = datasets[i][0].iloc[:,j-1]
            plt.scatter(zs[:,0], zs[:, 1], c=t, s=1)
            plt.title(titles[j-1])
        if save_name is None:
            plt.show()
        else:
            print('Saving figure as ' + save_name[i])
            plt.savefig(save_name[i])
            plt.show()
            plt.close

def get_imputed_values(model, all_data, head_id):
    imputed_values = []
    for i, batch in enumerate(all_data):
        imputed,_ ,_ = model(batch.float(), head_id)
        imputed_values.append(imputed)
    imputed = torch.cat(imputed_values).detach().cpu().numpy()
    return imputed

# +
def plot_imputed_values(trainer, latent2d, latents, datasets, titles, save_name=None):
    imputed1 = get_imputed_values(trainer.vae, datasets[0][5], 0)
    imputed2 = get_imputed_values(trainer.vae, datasets[1][5], 1)

    cms1 = ["Greens", "Blues", "Reds", "Oranges"]
    cms2 = ["Purples", "Greys", "YlOrBr", "GnBu"]
    
    #better way for multiple tubes
    for i in range(trainer.n_tubes):
        fig = plt.figure(figsize=(20,5))
        for j in range(1,5):
            zs = latent2d[i*latents[i].shape[0] : (i+1)*latents[i].shape[0]]
            plt.subplot(1,4,j)
            if i==0:
                t = datasets[i][0].iloc[:,j+2]
            else:
                t = imputed2[:, j+2]
            plt.scatter(zs[:,0], zs[:, 1], c=t, s=1, cmap=cms1[j-1])
            plt.title(titles[i][j-1])
    
    for i in range(trainer.n_tubes):
        fig = plt.figure(figsize=(20,5))
        for j in range(1,5):
            zs = latent2d[i*latents[i].shape[0] : (i+1)*latents[i].shape[0]]
            plt.subplot(1,4,j)
            if i==1:
                t = datasets[i][0].iloc[:,j+2]
            else:
                t = imputed1[:, j+6]
            plt.scatter(zs[:,0], zs[:, 1], c=t, s=1, cmap=cms2[j-1])
            plt.title(titles[i][j-1]) 
 

# new plotting function, plot only one set (4) of markers for all imputed + the one having actual measurements
def plot_imputed_values_set(trainer, tube_id, latent2d, latents, datasets, titles, save_name=None):
    n_tubes = trainer.n_tubes
    imputed = []
    tubes = []
    for i in range(n_tubes):
        if i != tube_id:
            imputed.append(get_imputed_values(trainer.vae, datasets[i][5], i))
            tubes.append(i)

    cms1 = ["Greens", "Blues", "Reds", "Oranges"]

    fig = plt.figure(figsize=(20,5))
    fig.suptitle("True values for tube %s"%(tube_id+1), fontsize=16)
    for k in range(1,5):
        zs = latent2d[tube_id*latents[tube_id].shape[0] : (tube_id+1)*latents[tube_id].shape[0]]
        plt.subplot(1,4,k)
        t = datasets[tube_id][0].iloc[:,2+k]
        plt.scatter(zs[:,0], zs[:, 1], c=t, s=1, cmap=cms1[k-1])
        plt.title(titles[k-1]) 
    if save_name is None:
        plt.show()
    else:
        print('Saving figure as ' + save_name[0])
        plt.savefig(save_name[0])
        plt.show()
        plt.close
    latent2d = np.delete(latent2d, np.s_[tube_id*latents[tube_id].shape[0] : (tube_id+1)*latents[tube_id].shape[0]], 0)
    for i in range(len(imputed)):
        fig = plt.figure(figsize=(20,5))
        fig.suptitle("Imputed values for tube %s"%(tubes[i]+1), fontsize=16)
        for j in range(1,5):
            zs = latent2d[i*latents[i].shape[0] : (i+1)*latents[i].shape[0]]
            plt.subplot(1,4,j)
            t = imputed[i][:, 2+tube_id*4+j]
            plt.scatter(zs[:,0], zs[:, 1], c=t, s=1, cmap=cms1[j-1])
            plt.title(titles[j-1])
        if save_name is None:
            plt.show()
        else:
            print('Saving figure as ' + save_name[i+1])
            plt.savefig(save_name[i+1])
            plt.show()
            plt.close

def create_colors_from_labels(labels):
    """
    Create cell colors from cell labels
    :param labels: cell labels
    :return: cell colors
    """
    # Set label colors
    unique_labels = np.unique(labels)
    n_labels = len(unique_labels)
    if n_labels <= 10:
        cell_colors = colors10(n_labels)
    elif n_labels <= 20:
        cell_colors = colors20(n_labels)
    else:
        raise ValueError("n_labels > 20, you must specify cell_colors yourself!")
    colors_dict = dict(zip(unique_labels, cell_colors))
    colors = [colors_dict[lab] for lab in labels]
    return np.array(colors)


class CellPlotter2D:
    """"
    Class for creating 2D plots.
    """
    def __init__(self, z, labels, colors=None, title="Title not set", xlab="x", ylab="y", figsize=(8, 8)):

        self.z = z
        self.labels = labels
        self.colors = colors if colors is not None else create_colors_from_labels(labels)
        self.figure_size = figsize
        self.font_size = 12
        self.basic_legend = True
        self.scatter_kwargs = dict(alpha=0.7)
        self.text_box_props = dict(boxstyle='round', facecolor='white', alpha=0.6)
        self.legend_labels = []
        self.title = title
        self.x_label = xlab
        self.y_label = ylab
        self.xlim = None
        self.ylim = None
        self.title_fontdict = dict(fontweight='bold')
        self.check_dimensions()

    def check_dimensions(self):
        """
        Dimensionality checks
        """
        assert self.z.shape[1] == 2, "z must have shape [n_cells, 2]"
        assert self.z.shape[0] == len(self.colors), "colors must have length equal to z.shape[0]"
        assert self.z.shape[0] == len(self.labels), "labels must have length equal to z.shape[0]"

    def draw_points(self):
        """
        Draw the cells
        """
        self.check_dimensions()
        for c in np.unique(self.colors):
            indices = np.where(self.colors == c)[0].tolist()
            i0 = indices[0]
            self.legend_labels += [self.labels[i0]]
            plt.scatter(self.z[indices, 0], self.z[indices, 1], color=c, **self.scatter_kwargs)

    def draw_label_boxes(self):
        """
        Draw cluster label text boxes
        """
        self.check_dimensions()
        for c in np.unique(self.colors):
            indices = np.where(self.colors == c)[0].tolist()
            i0 = indices[0]
            zz = np.median(self.z[indices, :], axis=0)
            plt.text(zz[0], zz[1], self.labels[i0], fontsize=self.font_size, bbox=self.text_box_props)

    def show(self, save_name):
        """
        Final stage of self.draw
        """

        # Set title, axis limits and labels
        plt.title(self.title, self.title_fontdict)
        plt.xlabel(self.x_label)
        plt.ylabel(self.y_label)
        if self.xlim is not None:
            plt.xlim(self.xlim)
        if self.ylim is not None:
            plt.ylim(self.ylim)

        # Saving
        if save_name is not None:
            print("Saving figure as " + save_name)
            plt.savefig(fname=save_name)
            plt.show()
            plt.close()
        else:
            plt.show()

    def draw(self, save_name=None):
        """
        Create the plot
        """
        fig = plt.figure(figsize=self.figure_size)
        self.draw_points()
        plt.legend(self.legend_labels) if self.basic_legend else self.draw_label_boxes()
        self.show(save_name)
