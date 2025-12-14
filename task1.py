# classifier_comparison_interactive.py
import plotly.graph_objects as go
import plotly.subplots as sp
from plotly.colors import qualitative
import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

def create_interactive_visualization():
    """创建交互式分类器比较可视化"""
    
    # 创建数据集
    X, y = make_classification(
        n_samples=300, n_features=2, n_informative=2, n_redundant=0,
        n_clusters_per_class=1, flip_y=0.1, random_state=42
    )
    
    # 定义分类器
    classifiers = {
        'Logistic Regression': LogisticRegression(),
        'Decision Tree': DecisionTreeClassifier(max_depth=5),
        'Random Forest': RandomForestClassifier(n_estimators=100),
        'SVM (RBF)': SVC(kernel='rbf', probability=True),
        'k-NN': KNeighborsClassifier(n_neighbors=5)
    }
    
    # 创建网格
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                         np.arange(y_min, y_max, 0.1))
    
    # 创建子图
    fig = sp.make_subplots(
        rows=2, cols=3,
        subplot_titles=list(classifiers.keys()) + ['Comparison'],
        specs=[[{'type': 'heatmap'}, {'type': 'heatmap'}, {'type': 'heatmap'}],
               [{'type': 'heatmap'}, {'type': 'heatmap'}, {'type': 'bar'}]]
    )
    
    accuracies = []
    
    # 训练每个分类器并创建热图
    for i, (name, clf) in enumerate(classifiers.items()):
        clf.fit(X, y)
        
        # 预测每个网格点
        Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
        Z = Z.reshape(xx.shape)
        
        # 计算准确率
        accuracy = clf.score(X, y)
        accuracies.append(accuracy)
        
        # 添加决策边界热图
        row = i // 3 + 1
        col = i % 3 + 1
        
        fig.add_trace(
            go.Heatmap(
                x=xx[0, :],
                y=yy[:, 0],
                z=Z,
                colorscale='RdBu',
                showscale=False,
                hoverinfo='none'
            ),
            row=row, col=col
        )
        
        # 添加数据点
        fig.add_trace(
            go.Scatter(
                x=X[:, 0],
                y=X[:, 1],
                mode='markers',
                marker=dict(
                    color=y,
                    colorscale=[[0, 'red'], [1, 'blue']],
                    size=8,
                    line=dict(width=1, color='black')
                ),
                showlegend=False,
                hoverinfo='skip'
            ),
            row=row, col=col
        )
        
        # 更新子图布局
        fig.update_xaxes(title_text="Feature 1", row=row, col=col)
        fig.update_yaxes(title_text="Feature 2", row=row, col=col)
        fig.layout.annotations[i].text = f"{name}<br>Accuracy: {accuracy:.3f}"
    
    # 添加准确率比较条形图
    fig.add_trace(
        go.Bar(
            x=list(classifiers.keys()),
            y=accuracies,
            marker_color=qualitative.Plotly[:len(accuracies)],
            text=[f'{acc:.3f}' for acc in accuracies],
            textposition='auto'
        ),
        row=2, col=3
    )
    
    fig.update_xaxes(title_text="Classifier", row=2, col=3)
    fig.update_yaxes(title_text="Accuracy", row=2, col=3)
    fig.layout.annotations[-1].text = "Classifier Accuracy Comparison"
    
    # 更新布局
    fig.update_layout(
        title_text="Interactive Classifier Comparison",
        height=800,
        showlegend=False
    )
    
    return fig

# 运行交互式可视化
if __name__ == "__main__":
    fig = create_interactive_visualization()
    fig.write_html("task1.html")
    print("交互式可视化已保存为 task1.html")
    fig.show()