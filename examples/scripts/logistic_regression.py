from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from celluloid import Camera
from tqdm import tqdm
from random import shuffle
import pylab as pl
import seaborn as sns
import pandas as pd
import numpy as np

# each animaiton class is made up from the actual sklearn Class
# this way I don't have to recreate the entire implementation
# and I'm able to call things like I'm used to (fit, score, etc.)
# and also I just have to create two new functions -> animate_training
# and plot_decision_boundary
class LogisticRegressionAnimation(LogisticRegression):
    """
    Creates a 2D Logistic Regression decision boundary animation 
    with all chosen parameters this class is inherted from sklearn, 
    so you may chose whatever you like with the init params
    """
    def __init__(self):
        super().__init__() # inheriting from LogisticRegression
        
    
    def plot_decision_boundary(self, x, y, fig, ax, train_size, cam, full_x, full_y):
        
        cm = plt.cm.RdBu # colormap
        cm_bright = ListedColormap(['#FF0000', '#0000FF']) # cm params
        x_min, x_max = full_x[:, 0].min() - .5, full_x[:, 0].max() + .5
        y_min, y_max = full_x[:, 1].min() - .5, full_x[:, 1].max() + .5
        xx, yy = np.meshgrid(np.arange(x_min, x_max, .05),
                                np.arange(y_min, y_max, .05))
        Z = self.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1] # creating the predicition proba for each spot in meshgrid
        Z = Z.reshape(xx.shape)
        ax.contourf(xx, yy, Z, cmap=cm, alpha=.8) # contourplotting
        ax.set_title("{:.3%} of the traning data used".format(len(x)/train_size))
        sns.scatterplot(full_x[:,0], full_x[:,1], hue=full_y, legend=False, palette=['#FF0000', '#0000FF'])

        cam.snap()
        
    def animate_training(self, x_train, y_train, x_test, y_test):
        """
        This method plots each iteration of the training process
        """
        # just shuffling so that the training is not the same every time you run it
        data = np.c_[x_train, y_train]
        df = pd.DataFrame(data)
        df_shuffled = df.sample(frac=1)
        shuffled_x_train = df_shuffled[[0, 1]].values
        shuffled_y_train = df_shuffled[[2]].values.ravel()
        
        # setting up celluloid 
        fig, ax = plt.subplots(figsize=(5,5))
        cam = Camera(fig)
        
        # tqdm makes a progressbar
        for i in tqdm(range(len(x_train))):
            try:
                self.fit(shuffled_x_train[:i+1], shuffled_y_train[:i+1])
                if i % 2 == 0 or i == len(x_train)-1:
                    self.plot_decision_boundary(shuffled_x_train[:i+1], shuffled_y_train[:i+1], fig, ax, len(x_train), cam, shuffled_x_train, shuffled_y_train)
            except:
                pass
        return cam.animate()
        

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import numpy as np
from sklearn.datasets import make_moons
x, y = make_moons(n_samples=400, noise=0.70)


x_train, x_test, y_train, y_test = train_test_split(\
                x, y, test_size=0.3, random_state=45)
anim = LogisticRegressionAnimation()
k = anim.animate_training(x_train, y_train, x_test, y_test)
k.save("..\\animations\\logistic_regression.mp4")