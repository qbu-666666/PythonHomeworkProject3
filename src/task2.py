# task2.py (3D多分类决策边界可视化)
import numpy as np
import plotly.graph_objects as go
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

# 加载数据，使用三个特征
iris = load_iris()
X = iris.data[:, [0, 2, 3]]  # sepal length, petal length, petal width
y = iris.target
feature_names = ['Sepal Length (cm)', 'Petal Length (cm)', 'Petal Width (cm)']
species = iris.target_names

# 划分训练/测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 构建3D网格（分辨率30，速度与平滑度平衡）
res = 30
x0 = np.linspace(X[:,0].min() - 0.5, X[:,0].max() + 0.5, res)
x1 = np.linspace(X[:,1].min() - 0.5, X[:,1].max() + 0.5, res)
x2 = np.linspace(X[:,2].min() - 0.5, X[:,2].max() + 0.5, res)
xx, yy, zz = np.meshgrid(x0, x1, x2)
grid = np.c_[xx.ravel(), yy.ravel(), zz.ravel()]

# 专业颜色（色盲友好）
colors = ['#2166ac', '#fdae61', '#5aae61']  # setosa: 深蓝, versicolor: 橙, virginica: 绿

# 创建图形
fig = go.Figure()

# 添加所有数据点（带黑边）
for c in range(3):
    mask = y == c
    fig.add_trace(go.Scatter3d(
        x=X[mask, 0], y=X[mask, 1], z=X[mask, 2],
        mode='markers',
        marker=dict(size=8, color=colors[c], line=dict(width=2, color='black')),
        name=species[c]
    ))

# 为每个类别训练 one-vs-rest RBF SVM，绘制决策曲面 (decision=0 的等值面)
for c in range(3):
    # 关键修复：使用 y_train 而不是完整的 y
    y_binary_train = (y_train == c).astype(int)
    
    svm = SVC(kernel='rbf', C=100, gamma='scale')
    svm.fit(X_train, y_binary_train)
    
    # 计算决策函数值
    decision = svm.decision_function(grid).reshape(xx.shape)
    
    # 绘制 P≈0.5 的决策曲面
    fig.add_trace(go.Isosurface(
        x=xx.flatten(),
        y=yy.flatten(),
        z=zz.flatten(),
        value=decision.flatten(),
        isomin=0,
        isomax=0,
        surface_count=1,
        colorscale=[[0, colors[c]], [1, colors[c]]],
        opacity=0.6,
        showscale=False,
        name=f'{species[c]} decision surface',
        caps=dict(x_show=False, y_show=False, z_show=False)
    ))

# 美化布局
fig.update_layout(
    title="<b>Task 2: 3D Multiclass Decision Boundaries (Iris Dataset)</b><br>"
          "<sup>Features: Sepal Length, Petal Length, Petal Width | Model: RBF SVM (one-vs-rest)<br>"
          "Three curved surfaces separate the three classes</sup>",
    title_x=0.5,
    height=900,
    width=1200,
    scene=dict(
        xaxis_title=feature_names[0],
        yaxis_title=feature_names[1],
        zaxis_title=feature_names[2],
        camera=dict(eye=dict(x=1.7, y=1.7, z=1.5)),
        aspectmode='data'
    ),
    margin=dict(l=50, r=50, t=100, b=50)
)

fig.write_html("task2.html")
print("task2.html 已成功生成！")