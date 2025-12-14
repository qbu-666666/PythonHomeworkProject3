# task4.py (高级交互式3D分类器可视化)
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF

# 数据加载：使用三个最具区分度的特征
iris = load_iris()
X = iris.data[:, [0, 2, 3]]  # sepal length, petal length, petal width
y = iris.target
feature_names = ['Sepal Length (cm)', 'Petal Length (cm)', 'Petal Width (cm)']
species = iris.target_names

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 高分辨率3D网格（res=35平衡速度与平滑度，可改为40更细腻）
res = 35
x0 = np.linspace(X[:,0].min()-0.5, X[:,0].max()+0.5, res)
x1 = np.linspace(X[:,1].min()-0.5, X[:,1].max()+0.5, res)
x2 = np.linspace(X[:,2].min()-0.5, X[:,2].max()+0.5, res)
xx, yy, zz = np.meshgrid(x0, x1, x2)
grid = np.c_[xx.ravel(), yy.ravel(), zz.ravel()]

# 专业颜色（色盲友好）
colors = ['#2166ac', '#fdae61', '#5aae61']  # 深蓝, 橙, 绿

# 使用Gaussian Process Classifier（支持多分类、平滑概率、非线性边界）
gpc = GaussianProcessClassifier(kernel=1.0 * RBF(1.0), random_state=42, n_jobs=-1)
gpc.fit(X_train, y_train)

# 预测概率（形状: (res, res, res, 3)）
probs = gpc.predict_proba(grid).reshape((res, res, res, 3))

# 硬决策（取概率最大的类别）
Z = np.argmax(probs, axis=3)

# 创建1行3列子图
fig = make_subplots(
    rows=1, cols=3,
    specs=[[{'type': 'scene'}, {'type': 'scene'}, {'type': 'scene'}]],
    subplot_titles=(
        "3D Decision Surfaces<br>(P=0.5 Isosurfaces - GPC)",
        "Probability Volume Rendering<br>(P(virginica))",
        "Interactive 3D Scatter<br>(Toggle classes in legend)"
    )
)

# === 子图1: 每个类别的决策面 (概率=0.5 等值面) ===
for c in range(3):
    fig.add_trace(go.Isosurface(
        x=xx.flatten(),
        y=yy.flatten(),
        z=zz.flatten(),
        value=probs[:,:,:,c].flatten(),
        isomin=0.5,
        isomax=0.5,
        surface_count=1,
        colorscale=[[0, colors[c]], [1, colors[c]]],
        opacity=0.7,
        showscale=False,
        name=f'{species[c]} boundary (P=0.5)',
        caps=dict(x_show=False, y_show=False, z_show=False)
    ), row=1, col=1)

# 添加数据点
for c in range(3):
    mask = y == c
    fig.add_trace(go.Scatter3d(
        x=X[mask,0], y=X[mask,1], z=X[mask,2],
        mode='markers',
        marker=dict(size=7, color=colors[c], line=dict(width=2, color='black')),
        name=species[c]
    ), row=1, col=1)

# === 子图2: 概率体积渲染 (以virginica为例) ===
fig.add_trace(go.Volume(
    x=xx.flatten(),
    y=yy.flatten(),
    z=zz.flatten(),
    value=probs[:,:,:,2].flatten(),  # virginica概率
    isomin=0.05,
    isomax=0.95,
    opacity=0.08,
    opacityscale="uniform",
    surface_count=25,
    colorscale='Viridis',
    colorbar=dict(title="P(virginica)", thickness=15, len=0.6),
    caps=dict(x_show=False, y_show=False, z_show=False),
    name='P(virginica)'
), row=1, col=2)

# 数据点叠加
for c in range(3):
    mask = y == c
    fig.add_trace(go.Scatter3d(
        x=X[mask,0], y=X[mask,1], z=X[mask,2],
        mode='markers',
        marker=dict(size=6, color=colors[c], line=dict(width=1, color='black')),
        name=species[c],
        showlegend=False
    ), row=1, col=2)

# === 子图3: 纯交互式3D散点图（可开关类别）===
for c in range(3):
    mask = y == c
    fig.add_trace(go.Scatter3d(
        x=X[mask,0], y=X[mask,1], z=X[mask,2],
        mode='markers',
        marker=dict(size=8, color=colors[c], line=dict(width=2, color='black')),
        name=species[c]
    ), row=1, col=3)

# 统一美观布局
fig.update_layout(
    height=900,
    width=1900,
    title_text="<b>Advanced Interactive 3D Visualization of Multiclass Classification (Iris Dataset)</b><br>"
               "<sup>Model: Gaussian Process Classifier | Features: Sepal Length, Petal Length, Petal Width | "
               "Rotate, zoom, hover, and toggle layers via legend</sup>",
    title_x=0.5,
    font=dict(family="Arial", size=14),
    scene=dict(
        xaxis_title=feature_names[0],
        yaxis_title=feature_names[1],
        zaxis_title=feature_names[2],
        camera=dict(eye=dict(x=1.8, y=1.8, z=1.5))
    ),
    scene2=dict(
        xaxis_title=feature_names[0],
        yaxis_title=feature_names[1],
        zaxis_title=feature_names[2],
        camera=dict(eye=dict(x=1.8, y=1.8, z=1.5))
    ),
    scene3=dict(
        xaxis_title=feature_names[0],
        yaxis_title=feature_names[1],
        zaxis_title=feature_names[2],
        camera=dict(eye=dict(x=1.8, y=1.8, z=1.5))
    ),
    margin=dict(l=50, r=50, t=120, b=50)
)

fig.write_html("task4.html")
print("task4.html 已成功生成！")