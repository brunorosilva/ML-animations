import numpy as np
import matplotlib.pyplot as plt
from celluloid import Camera
from sklearn.svm import SVC

# Adaptado do material de  R. Jordan Crouser do Smith College (2016)
def anim_svc(svc, X, y, h=0.05, pad=0.25):
    x_min, x_max = X[:, 0].min()-pad, X[:, 0].max()+pad
    y_min, y_max = X[:, 1].min()-pad, X[:, 1].max()+pad
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = svc.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    plt.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.2)

    plt.scatter(X[:,0], X[:,1], s=70, c=y, cmap=plt.cm.Paired)
    # Mostra os vetores de suporte com um 'x'
    sv = svc.support_vectors_
    plt.scatter(sv[:,0], sv[:,1], c='k', marker='x', s=100, linewidths='1')
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.xlabel('X1')
    plt.ylabel('X2')
    cam.snap()
    return cam

fig = plt.figure(figsize=(10, 5))
cam = Camera(fig)

x = np.random.uniform(-4*np.pi, 4*np.pi, size=2000)
y = np.random.uniform(-2, 2, size=2000)
z = [1 if yy>np.sin(xx) else -1 for xx, yy in zip(x, y)]

j = 0
for c in [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000]:
  
  model = SVC(c).fit(np.array([x, y]).T, z)
  anim = anim_svc(model, np.array([x, y]).T, z)

animation = anim.animate()
animation.save("SVC.gif")
