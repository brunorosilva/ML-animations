import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from celluloid import Camera
from sklearn.datasets import make_regression
import tensorflow.compat.v1 as tf
from IPython.display import HTML
import matplotlib
#pylint: disable=E1120 

__author__ = "Bruno Rodrigues Silva"

st.title("Sim")


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
    
    def fit(self,
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
        camera = Camera(fig)
        
        with tf.Session() as sess: # sessão de treinamento

            sess.run(init)

            for epoch in range(epochs): # processo de treinamento
                for (x, y) in zip(self.train_X, self.train_Y):
                    sess.run(optimizer, feed_dict={self.X: x, self.Y: y}) # rodar cenários de treinamento

                if epoch % 10 == 0 or epoch == 0:
                    c = sess.run(cost, feed_dict={self.X:self.train_X, self.Y:self.train_Y})
                    w_atual = sess.run(self.W)
                    b_atual = sess.run(self.b)
                    eq_reta = 'Y = ' + str(w_atual) + ' * X + ' + str(b_atual) # equacao da reta atual

                    #a = st.markdown("# Época: " + "{:04d}".format(epoch+1) + "\ncusto=" + "{:.9f}".format(c) + "\n\nW=" + str(w_atual) + "\n\nb=" + str(b_atual)) # vamos ver como estão os indicadores
                    #a = st.markdown('')
                    if line_equation_legend:
                        linha, = ax.plot(self.train_X, sess.run(self.W) * self.train_X + sess.run(self.b), c=line_color, label=eq_reta)
                    else:
                        linha, = ax.plot(self.train_X, sess.run(self.W) * self.train_X + sess.run(self.b), c=line_color)
                    pontos = ax.scatter(self.train_X, self.train_Y, c=dot_color, label='Dados de treino') # dados de treino
                    if rmse_on_title:
                        ax.text(0.5, 1.01, "RMSE = "+ str(np.sqrt(c)), transform=ax.transAxes) # fazendo o título

                    plt.legend()
                    camera.snap()
                    pontos.set_label('')
                    linha.set_label('')
                    

            
            st.markdown("Optimização terminada!")
            training_cost = sess.run(cost, feed_dict={self.X: self.train_X, self.Y: self.train_Y})
            st.markdown("Custo=" + str(training_cost) + " W=" + str(sess.run(self.W)) + " b=" + str(sess.run(self.b)) + '\n')
            animation = camera.animate()
            animation.save('uhu.mp4')
            v = open('uhu.mp4', 'rb')
            st.video(v.read())




#dataset = st.sidebar.selectbox("Select your dataset right here")
w = st.sidebar.slider("W", -100, 100, 1)
b = st.sidebar.slider("b", -100, 100, 1)
alpha = st.sidebar.slider("alpha", 0.0001, 10., 0.1, 0.01)
size = st.sidebar.slider("size", 10, 5000, 50)
noise = st.sidebar.slider("noise", 0, 50, 1)
epochs = st.sidebar.slider("epochs", 1, 10000, 100)
see_rmse = st.sidebar.checkbox("Show RMSE on Title")
see_line_eq = st.sidebar.checkbox("Show Line equation")

clist = ['blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'black', 'white']
line_c = st.sidebar.selectbox("Line Color", clist, 6)
dots_c = st.sidebar.selectbox("Dots Color", clist, 2)


if st.checkbox("Fit Model"):
    LinearRegressionAnimation(W=w, b = b, size=size, alpha=alpha, noise=noise).fit(epochs=epochs, line_color=line_c, dot_color=dots_c, rmse_on_title=see_rmse, line_equation_legend=see_line_eq)
else:
    LinearRegressionAnimation(W=w, b = b, size=size, alpha=alpha, noise=noise).fit(epochs=1, line_color=line_c, dot_color=dots_c, rmse_on_title=see_rmse, line_equation_legend=see_line_eq)
    