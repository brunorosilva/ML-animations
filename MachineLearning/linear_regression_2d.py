# imports necessários

import matplotlib.pyplot as plt
import numpy as np
import tensorflow.compat.v1 as tf
from celluloid import Camera
from sklearn.datasets import make_regression

tf.disable_v2_behavior()
class LinearRegressionAnimation():
    """
    Esse objeto cria um arquivo mp4, GIF ou um vídeo HTML5 do algoritmo de Regressão Linear em 2D
    This object creates a mp4, GIF or HTML5 video of the Linear Regression Algorithm in 2D
    """        
        
    def __init__(
        self, train_X=np.nan, train_Y=np.nan, W = np.random.randn(), b = np.random.randn(), 
        alpha=0.01, size=100, noise=2):
        """
        Você pode escolher os dados x, y de treino, w (inclinação da reta), b (parâmetro independente),
        alpha (velocidade de treinamento), size (tamanho do dataset aleatório de treino), noise (barulho no treino aleatório),
        se não quiser escolher nenhum desses parâmetros, serão utilizados dados aleatórios
        """
        if np.isnan(train_X): # se não forem passados dados de treino, usasse o aleatório
            print("Taking random X")
            self.train_X, self.train_Y = make_regression(size, 1, noise=noise)
        else:
            self.train_X = train_X
            self.train_Y = train_Y
        self.X = tf.placeholder("float") # criando as variáveis tf placeholder com orientação de grafo
        self.Y = tf.placeholder("float")

        self.W = tf.Variable(float(W), name="weight")
        self.b = tf.Variable(float(b), name="bias")
        self.alpha = alpha
        self.n_samples = size
    
    def animate(self,
        frames=100,
        line_color='black', 
        dot_color='red',
        epochs=1000, 
        plot_size=(9,6),
        rmse_on_title=True,
        line_equation_legend=True):
    
        """
        Esse método anima os dados passados no momento do init, você pode escolher quantos frames seu video vai ter,
        tamanho do gráfico, rmse no título e equação da reta na legenda do gráfico
        """

        pred = self.X * self.W + self.b # equacao da reta

        cost = (tf.reduce_sum(tf.pow(pred-self.Y, 2))/(2*self.n_samples)) # equacao de custo => erro quadrático médio
        
        optimizer = tf.train.GradientDescentOptimizer(self.alpha).minimize(cost) # optimizador => gradiente descendente estocástico

        init = tf.global_variables_initializer() # iniciando o grafo

        fig, ax = plt.subplots(figsize=plot_size) # criando a figura
        cam = Camera(fig) # criando a camera que vai tirar fotos do gráfico

        with tf.Session() as sess: # sessão de treinamento

            sess.run(init)

            for epoch in range(epochs): # processo de treinamento
                for (x, y) in zip(self.train_X, self.train_Y):
                    sess.run(optimizer, feed_dict={self.X: x, self.Y: y}) # rodar cenários de treinamento

                if ((epoch + 1) % (epochs/frames) == 0) or (epoch + 1 == epochs) or epoch == 0: # vemos a primeira, as necessárias pro número de frames e a última iterações
                    c = sess.run(cost, feed_dict={self.X:self.train_X, self.Y:self.train_Y})
                    w_atual = sess.run(self.W)
                    b_atual = sess.run(self.b)
                    eq_reta = 'Y = ' + str(w_atual) + ' * X + ' + str(b_atual) # equacao da reta atual

                    print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(c),
                        "W=", str(w_atual), "b=", str(b_atual)) # vamos ver como estão os indicadores
                    
                    linha, = ax.plot(self.train_X, sess.run(self.W) * self.train_X + sess.run(self.b), c=line_color, label=eq_reta)
                    pontos = ax.scatter(self.train_X, self.train_Y, c=dot_color, label='Dados de treino') # dados de treino
                    ax.text(0.5, 1.01, "RMSE = "+ str(np.sqrt(c)), transform=ax.transAxes) # fazendo o título

                    plt.legend()
                    cam.snap()
                    pontos.set_label('')
                    linha.set_label('')

            anim = cam.animate()
            
            print("Optimization Finished!")
            training_cost = sess.run(cost, feed_dict={self.X: self.train_X, self.Y: self.train_Y})
            print("Training cost=", training_cost, "W=", sess.run(self.W), "b=", sess.run(self.b), '\n')

        return anim
    

ling = LinearRegressionAnimation(W=-90, b = 150)
animation = ling.animate()
animation.save("Linear_Regression.mp4")
