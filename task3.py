# task3.py
import numpy as np
import plotly.graph_objects as go
from sklearn.datasets import load_iris
from sklearn.svm import SVC

iris = load_iris()
mask = iris.target > 0
X = iris.data[mask, :3]
y = iris.target[mask] - 1

clf = SVC(kernel='rbf', probability=True, gamma='scale')
clf.fit(X, y)

res = 25
xx, yy, zz = np.meshgrid(np.linspace(X[:,0].min()-1, X[:,0].max()+1, res),
                         np.linspace(X[:,1].min()-1, X[:,1].max()+1, res),
                         np.linspace(X[:,2].min()-1, X[:,2].max()+1, res))
grid = np.c_[xx.ravel(), yy.ravel(), zz.ravel()]
probs = clf.predict_proba(grid)[:,1].reshape(xx.shape)  # 类别1的概率

fig = go.Figure()

# 数据点
fig.add_trace(go.Scatter3d(x=X[y==0,0], y=X[y==0,1], z=X[y==0,2],
                           mode='markers', marker=dict(color='green', size=6), name='Versicolor'))
fig.add_trace(go.Scatter3d(x=X[y==1,0], y=X[y==1,1], z=X[y==1,2],
                           mode='markers', marker=dict(color='blue', size=6), name='Virginica'))

# 概率体积渲染（透明度表示概率）
fig.add_trace(go.Volume(
    x=xx.flatten(), y=yy.flatten(), z=zz.flatten(),
    value=probs.flatten(),
    isomin=0.1, isomax=0.9,
    opacity=0.1,
    opacityscale="uniform",
    surface_count=15,
    colorscale='Viridis',
    caps=dict(x_show=False, y_show=False, z_show=False),
    name='Probability Volume'
))

# 决策面 (prob=0.5)
fig.add_trace(go.Isosurface(
    x=xx.flatten(), y=yy.flatten(), z=zz.flatten(),
    value=probs.flatten(),
    isomin=0.5, isomax=0.5,
    surface_fill=1.0,
    colorscale=[[0,'gray'],[1,'gray']],
    showscale=False,
    name='Decision Surface (p=0.5)'
))

fig.update_layout(title="Task 3: 3D Probability Map (Binary Classification, 3 Features)",
                  scene=dict(xaxis_title='Sepal Length', yaxis_title='Petal Length', zaxis_title='Petal Width'),
                  height=900)
fig.write_html("task3.html")
print("task3.html generated successfully!")