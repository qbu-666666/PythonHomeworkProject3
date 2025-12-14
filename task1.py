# task1.py
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

# 数据加载
iris = load_iris()
X = iris.data[:, 2:]  # petal length & petal width
y = iris.target
species = ['setosa', 'versicolor', 'virginica']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 分类器
classifiers = {
    'Logistic Regression': LogisticRegression(max_iter=200, multi_class='multinomial', random_state=42),
    'Linear SVM': SVC(kernel='linear', probability=True, random_state=42),
    'Decision Tree (depth=5)': DecisionTreeClassifier(max_depth=5, random_state=42)
}

# 高分辨率网格
x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 500),
                     np.linspace(y_min, y_max, 500))
grid = np.c_[xx.ravel(), yy.ravel()]

# 专业颜色
class_colors = ['#2166ac', '#fdae61', '#5aae61']  # 深蓝, 橙, 绿 (colorblind-safe, 科研常用)

# specs: 每行 5 列 (决策 + 3概率 + Max Class)
row_spec = [{"type": "xy"}, {"type": "contour"}, {"type": "contour"}, {"type": "contour"}, {"type": "xy"}]
specs = [row_spec.copy() for _ in range(3)]

# 子图标题
subplot_titles = []
for name in classifiers:
    subplot_titles += [f"{name}<br>Decision Regions", "P(setosa)", "P(versicolor)", "P(virginica)", "Max Class"]

fig = make_subplots(rows=3, cols=5,
                    subplot_titles=subplot_titles,
                    specs=specs,
                    horizontal_spacing=0.05,
                    vertical_spacing=0.12)

row = 1
for name, clf in classifiers.items():
    clf.fit(X_train, y_train)
    Z = clf.predict(grid).reshape(xx.shape)  # 硬决策
    has_proba = hasattr(clf, "predict_proba")
    if has_proba:
        probs = clf.predict_proba(grid).reshape((500, 500, 3))

    # 左侧: Decision Regions + 数据点
    for c in range(3):
        fig.add_trace(go.Scatter(x=X[y==c,0], y=X[y==c,1], mode='markers',
                                 marker=dict(color=class_colors[c], size=10, line=dict(width=2, color='black')),
                                 name=species[c], showlegend=(row==1)), row=row, col=1)
    fig.add_trace(go.Contour(z=Z, x=np.linspace(x_min, x_max, 500), y=np.linspace(y_min, y_max, 500),
                             colorscale=[[0,class_colors[0]], [0.5,class_colors[1]], [1,class_colors[2]]],
                             opacity=0.4, showscale=False, contours_coloring='fill', line_width=0),
                  row=row, col=1)

    # 中间3列: 每类概率热图 (白 → 该类颜色)
    if has_proba:
        for c in range(3):
            col_idx = c + 2
            cmap = [[0, 'white'], [1, class_colors[c]]]
            fig.add_trace(go.Contour(z=probs[:,:,c], x=np.linspace(x_min, x_max, 500), y=np.linspace(y_min, y_max, 500),
                                     colorscale=cmap, contours=dict(coloring='fill', showlabels=False),
                                     showscale=(row==1 and col_idx==4),  # 只在第一行最右概率显示colorbar
                                     colorbar=dict(title="Probability", thickness=12, len=0.6, x=1.02 if col_idx==4 else None)),
                          row=row, col=col_idx)
    else:
        for c in range(3):
            col_idx = c + 2
            fig.add_annotation(text="No probability<br>estimates", x=0.5, y=0.5, xref="x domain", yref="y domain",
                               showarrow=False, font=dict(size=15, color="gray"), row=row, col=col_idx)

    # 右侧: Max Class (等同于硬决策区域)
    for c in range(3):
        fig.add_trace(go.Scatter(x=X[y==c,0], y=X[y==c,1], mode='markers',
                                 marker=dict(color=class_colors[c], size=10, line=dict(width=2, color='black')),
                                 showlegend=False), row=row, col=5)
    fig.add_trace(go.Contour(z=Z, x=np.linspace(x_min, x_max, 500), y=np.linspace(y_min, y_max, 500),
                             colorscale=[[0,class_colors[0]], [0.5,class_colors[1]], [1,class_colors[2]]],
                             opacity=0.5, showscale=False, contours_coloring='fill', line_width=2),
                  row=row, col=5)

    row += 1

# 科研级美化布局
fig.update_layout(
    height=1100, width=1800,
    title_text="<b>Multiclass Classifier Decision Boundaries and Probability Maps on Iris Dataset</b><br>"
               "<sup>Petal Length vs. Petal Width | Left & Right: Decision Regions (filled) with true labels | Middle: Per-class posterior probabilities</sup>",
    title_x=0.5,
    font=dict(family="Arial", size=14),
    plot_bgcolor='white',
    paper_bgcolor='white',
    margin=dict(l=60, r=100, t=140, b=80)
)

# 统一坐标轴 & 网格
for r in range(1,4):
    for c in [1,5]:
        fig.update_xaxes(title_text="Petal Length (cm)" if r==3 else "", row=r, col=c, showgrid=True, gridcolor='lightgray')
        fig.update_yaxes(title_text="Petal Width (cm)" if c==1 else "", row=r, col=c, showgrid=True, gridcolor='lightgray')
    for c in range(2,5):
        fig.update_xaxes(showgrid=True, gridcolor='lightgray', zeroline=False, row=r, col=c)
        fig.update_yaxes(showgrid=True, gridcolor='lightgray', zeroline=False, row=r, col=c)

fig.write_html("task1.html")
print("task1.html 已生成！")