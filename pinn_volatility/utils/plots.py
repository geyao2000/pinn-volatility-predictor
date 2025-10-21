
import matplotlib.pyplot as plt
def plot_series(y_true, y_pred=None, title='Series', savepath=None):
    plt.figure()
    plt.plot(y_true, label='true')
    if y_pred is not None:
        plt.plot(y_pred, label='pred')
    plt.title(title); plt.legend()
    if savepath: plt.savefig(savepath, bbox_inches='tight')
    else: plt.show()
