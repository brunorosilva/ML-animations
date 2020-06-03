import matplotlib.pyplot as plt
import numpy as np
from celluloid import Camera
import tensorflow.compat.v1 as tf
from sklearn.datasets import make_regression

tf.disable_v2_behavior()
class LinearRegressionAnimation():
    """
    This object creates a mp4, GIF or HTML5 video of the Linear Regression Algorithm
    """        
        
    def __init__(
        self, train_X=np.nan, train_Y=np.nan, W = np.random.randn(), b = np.random.randn(), 
        alpha=0.01, size=100, noise=2):
        if np.isnan(train_X):
            print("Taking random X")
            self.train_X, self.train_Y = make_regression(size, 1, noise=noise)
        else:
            self.train_X = train_X
            self.train_Y = train_Y
        self.X = tf.placeholder("float")
        self.Y = tf.placeholder("float")

        self.W = tf.Variable(float(W), name="weight")
        self.b = tf.Variable(float(b), name="bias")
        self.alpha = alpha
        self.n_samples = size
    
    def animate(self,
        frames=20,
        line_color='black', 
        dot_color='red',
        epochs=1000, 
        plot_size=(9,6)):
    

        pred = self.X * self.W + self.b

        cost = tf.reduce_sum(tf.pow(pred-self.Y, 2))/(2*self.n_samples)
        
        optimizer = tf.train.GradientDescentOptimizer(self.alpha).minimize(cost)

        init = tf.global_variables_initializer()

        fig, ax = plt.subplots(figsize=plot_size)
        cam = Camera(fig)

        with tf.Session() as sess:

            sess.run(init)

            for epoch in range(epochs):
                for (x, y) in zip(self.train_X, self.train_Y):
                    sess.run(optimizer, feed_dict={self.X: x, self.Y: y})

                if (epoch + 1) % 10 == 0:
                    c = sess.run(cost, feed_dict={self.X:self.train_X, self.Y:self.train_Y})
                    print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(c), \
                        "W=", sess.run(self.W), "b=", sess.run(self.b))
                    
                    ax.scatter(self.train_X, self.train_Y, c=dot_color)
                    ax.plot(self.train_X, sess.run(self.W) * self.train_X + sess.run(self.b), c=line_color)
                    cam.snap()

            anim = cam.animate()
            
            print("Optimization Finished!")
            training_cost = sess.run(cost, feed_dict={self.X: self.train_X, self.Y: self.train_Y})
            print("Training cost=", training_cost, "W=", sess.run(self.W), "b=", sess.run(self.b), '\n')

        return anim


ling = LinearRegressionAnimation(W=-90, b = 50)
animation = ling.animate()
animation.save("Linear_Regression.mp4")