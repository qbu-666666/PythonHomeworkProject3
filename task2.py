# task2.py
import numpy as np
import plotly.graph_objects as go
from sklearn.datasets import load_iris
from sklearn.svm import SVC

iris = load_iris()
# 二分类：versicolor (1) vs virginica (2)
mask = iris.target > 0
X = iris.data[mask, :3]  # sepal_length, petal_length, petal_width
y = iris.target[mask] - 1  # 0: versicolor, 1: virginica

clf = SVC(kernel='linear')
clf.fit(X, y)

# 创建网格
def meshgrid3d(res=30):
    x_min, x_max = X[:,0].min()-1, X[:,0].max()+1
    y_min, y_max = X[:,1].min()-1, X[:,1].max()+1
    z_min, z_max = X[:,2].min()-1, X[:,2].max()+1
    xx, yy, zz = np.meshgrid(np.linspace(x_min, x_max, res),
                             np.linspace(y_min, y_max, res),
                             np.linspace(z_min, z_max, res))
    return xx, yy, zz

xx, yy, zz = meshgrid3d(20)
grid = np.c_[xx.ravel(), yy.ravel(), zz.ravel()]
Z = clf.predict(grid).reshape(xx.shape)

# 决策超平面：w·x + b = 0
w = clf.coef_[0]
b = clf.intercept_[0]
# 求解一个坐标轴上的值，例如 x0 = -(b + w1*x1 + w2*x2)/w0
x0_grid = (-b - w[1]*xx - w[2]*yy) / w[0]

fig = go.Figure()

# 数据点
fig.add_trace(go.Scatter3d(x=X[:,0], y=X[:,1], z=X[:,2],
                           mode='markers',
                           marker=dict(color=y, colorscale=['green','blue'], size=6),
                           name='Data'))

# 决策平面
fig.add_trace(go.Surface(x=xx[:,:,0], y=yy[:,:,0], z=x0_grid,
                         opacity=0.6, colorscale='gray', showscale=False, name='Decision Hyperplane'))

fig.update_layout(title="Task 2: 3D Decision Boundary (Binary Classification, 3 Features)",
                  scene=dict(xaxis_title='Sepal Length', yaxis_title='Petal Length', zaxis_title='Petal Width'),
                  height=800)
fig.write_html("task2.html")
print("task2.html generated successfully!")