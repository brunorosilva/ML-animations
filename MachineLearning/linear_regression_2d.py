import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from celluloid import Camera
import tensorflow as tf

class LinearRegressionAnimation():


    """
    This object creates a mp4, GIF or HTML5 video of the Linear Regression Algorithm
    """
    def __init__(self, X, Y, W = tf.Variable(np.random.randn(), name="weight"), b = tf.Variable(np.random.randn(), name="bias")):
        self.X          = X
        self.Y          = Y
        self.W          = W
        self.b          = b
        
    def linear_regression(self, x):
        return self.W * x + self.b
    
    def mean_square(self, y_pred, y_true):
        return tf.reduce_mean(tf.square(y_pred - y_true))

    def run_optimization(self):
        # Wrap computation inside a GradientTape for automatic differentiation.
        with tf.GradientTape() as g:
            pred = linear_regression(X)
            loss = mean_square(pred, Y) 

        # Compute gradients.
        gradients = g.gradient(loss, [W, b])
        
        # Update W and b following gradients.
        tf.keras.optimizers.SGD(learning_rate).apply_gradients(zip(gradients, [W, b]))




    def create_training_animation(
        self,
        frames=20,
        line_color='black', 
        dot_color='red', 
        alpha=0.01, 
        epochs=100, 
        plot_size=(9,6), 
        rmse_on_title=False,
        extension='mp4'):


        fig, ax = plt.subplots(figsize=plot_size) # criando o plot vazio
        camera = Camera(fig)


        for _ in range(epochs):
            y_pred = self.theta_1 + self.theta_2 * self.x
            error = y_pred - self.y
            mean_sq_er = np.sum(error**2)
            mean_sq_er = mean_sq_er / len(self.y)
            self.theta_1 = self.theta_1 - alpha * 2 * np.sum(error)/len(self.y)
            self.theta_2 = self.theta_2 - alpha * 2 * np.sum(error * self.x)/len(self.y)


            #ax.plot(self.x, y_pred)
            #ax.legend(j)
            ax.text(0.5, 1.01, "ok", transform=ax.transAxes) # fazendo o título
            print(mean_sq_er)
            #print(y_pred)
            camera.snap() # a foto da câmera
        anim = camera.animate() # sequenciando as fotos
        return anim

    def save_anim(self, animation_name="animation", extension='mp4'):
        self.anim.save(animation_name + "." + extension)


    def show_anim(self):
        pass





x = [1,2,3,4,5,6,7,8]
y = [2,4,5,7,9,10,11,12]

an = LinearRegressionAnimation(x, y)
an.create_training_animation()

        



























