# single_3d_probability_html.py
import numpy as np
import plotly.graph_objects as go
import plotly.subplots as sp
from plotly.subplots import make_subplots
from sklearn.datasets import make_classification, make_blobs, load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

def create_single_html_with_all_visualizations():
    """创建包含所有3D概率图的可视化HTML文件"""
    
    print("生成数据集...")
    # 生成多个3D数据集
    datasets = []
    
    # 1. 二分类3D数据集
    X1, y1 = make_classification(
        n_samples=200, n_features=3, n_informative=3, n_redundant=0,
        n_classes=2, n_clusters_per_class=1, random_state=42
    )
    datasets.append(("Binary 3D", X1, y1, 2))
    
    # 2. 三分类3D数据集
    X2, y2 = make_blobs(
        n_samples=200, centers=3, n_features=3,
        cluster_std=1.2, random_state=42
    )
    datasets.append(("3-Class 3D", X2, y2, 3))
    
    # 3. Iris数据集（3个特征）
    iris = load_iris()
    X3 = iris.data[:, :3]
    y3 = iris.target
    datasets.append(("Iris 3D", X3, y3, 3))
    
    # 定义分类器
    classifiers = [
        ("Logistic Regression", LogisticRegression(max_iter=1000)),
        ("Decision Tree", DecisionTreeClassifier(max_depth=5)),
        ("Random Forest", RandomForestClassifier(n_estimators=100)),
        ("SVM (RBF)", SVC(kernel='rbf', probability=True)),
        ("k-NN", KNeighborsClassifier(n_neighbors=5))
    ]
    
    # 创建主图形
    fig = make_subplots(
        rows=5, cols=4,
        specs=[
            [{'type': 'scene'}, {'type': 'scene'}, {'type': 'scene'}, {'type': 'scene'}],  # 第一行
            [{'type': 'scene'}, {'type': 'scene'}, {'type': 'scene'}, {'type': 'scene'}],  # 第二行
            [{'type': 'scene'}, {'type': 'scene'}, {'type': 'scene'}, {'type': 'scene'}],  # 第三行
            [{'type': 'scene'}, {'type': 'scene'}, {'type': 'scene'}, {'type': 'scene'}],  # 第四行
            [{'type': 'scene'}, {'type': 'scene'}, {'type': 'scene'}, {'type': 'scene'}]   # 第五行
        ],
        subplot_titles=[
            # 第一行：数据集概览
            "Dataset 1: Binary Classification", "Dataset 2: 3-Class", "Dataset 3: Iris", "All Datasets",
            # 第二行：Logistic Regression概率图
            "LR: Probability Isosurface", "LR: Class 0 Probability", "LR: Class 1 Probability", "LR: Decision Boundary",
            # 第三行：Random Forest概率图
            "RF: Probability Isosurface", "RF: Class 0 Probability", "RF: Class 1 Probability", "RF: Uncertainty Map",
            # 第四行：SVM概率图
            "SVM: Probability Isosurface", "SVM: Class 0 Probability", "SVM: Class 1 Probability", "SVM: Decision Regions",
            # 第五行：比较视图
            "All Classifiers Comparison", "Probability Slice (XY)", "Probability Slice (XZ)", "Probability Slice (YZ)"
        ],
        vertical_spacing=0.05,
        horizontal_spacing=0.02,
        row_heights=[0.2, 0.2, 0.2, 0.2, 0.2]
    )
    
    print("创建可视化...")
    
    # 颜色定义
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#F7DC6F', '#BB8FCE']
    
    # === 第一行：数据集概览 ===
    for col, (dataset_name, X, y, n_classes) in enumerate(datasets[:3], 1):
        # 标准化数据
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # 为每个类别添加数据点
        for class_idx in range(n_classes):
            class_mask = (y == class_idx)
            fig.add_trace(
                go.Scatter3d(
                    x=X_scaled[class_mask, 0],
                    y=X_scaled[class_mask, 1],
                    z=X_scaled[class_mask, 2],
                    mode='markers',
                    marker=dict(
                        size=5,
                        color=colors[class_idx],
                        opacity=0.8
                    ),
                    name=f'{dataset_name} - Class {class_idx}',
                    showlegend=False
                ),
                row=1, col=col
            )
        
        # 设置场景
        fig.update_scenes(
            dict(
                xaxis_title='Feature 1',
                yaxis_title='Feature 2',
                zaxis_title='Feature 3',
                camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
            ),
            row=1, col=col
        )
    
    # 第四列：所有数据集合并视图
    all_data = []
    all_labels = []
    offsets = [(-3, 0, 0), (0, 0, 0), (3, 0, 0)]
    
    for idx, (dataset_name, X, y, n_classes) in enumerate(datasets[:3]):
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # 添加偏移以便区分
        X_offset = X_scaled + offsets[idx]
        
        for class_idx in range(n_classes):
            class_mask = (y == class_idx)
            fig.add_trace(
                go.Scatter3d(
                    x=X_offset[class_mask, 0],
                    y=X_offset[class_mask, 1],
                    z=X_offset[class_mask, 2],
                    mode='markers',
                    marker=dict(
                        size=4,
                        color=colors[class_idx],
                        opacity=0.6
                    ),
                    name=f'Dataset {idx+1} - Class {class_idx}',
                    showlegend=False
                ),
                row=1, col=4
            )
    
    fig.update_scenes(
        dict(
            xaxis_title='Feature 1',
            yaxis_title='Feature 2',
            zaxis_title='Feature 3',
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
        ),
        row=1, col=4
    )
    
    # === 第二行：Logistic Regression概率图 ===
    # 使用第一个数据集
    dataset_name, X, y, n_classes = datasets[0]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # 训练Logistic Regression
    lr = LogisticRegression(max_iter=1000)
    lr.fit(X_train, y_train)
    accuracy = lr.score(X_test, y_test)
    
    # 创建网格
    grid_size = 20
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    z_min, z_max = X[:, 2].min() - 1, X[:, 2].max() + 1
    
    xx, yy, zz = np.meshgrid(
        np.linspace(x_min, x_max, grid_size),
        np.linspace(y_min, y_max, grid_size),
        np.linspace(z_min, z_max, grid_size)
    )
    
    # 预测概率
    grid_points = np.c_[xx.ravel(), yy.ravel(), zz.ravel()]
    proba = lr.predict_proba(grid_points)
    
    # 1. 概率等值面
    proba_class0 = proba[:, 0].reshape(xx.shape)
    
    fig.add_trace(
        go.Isosurface(
            x=xx.flatten(),
            y=yy.flatten(),
            z=zz.flatten(),
            value=proba_class0.flatten(),
            isomin=0.3,
            isomax=0.7,
            surface_count=3,
            colorscale='RdBu',
            opacity=0.3,
            showscale=False,
            caps=dict(x_show=False, y_show=False, z_show=False)
        ),
        row=2, col=1
    )
    
    # 添加数据点
    for class_idx in range(n_classes):
        class_mask = (y == class_idx)
        fig.add_trace(
            go.Scatter3d(
                x=X[class_mask, 0],
                y=X[class_mask, 1],
                z=X[class_mask, 2],
                mode='markers',
                marker=dict(
                    size=5,
                    color=colors[class_idx],
                    opacity=0.8
                ),
                showlegend=False
            ),
            row=2, col=1
        )
    
    # 2. Class 0概率热图
    proba_slice_z = proba_class0[:, :, grid_size//2]
    fig.add_trace(
        go.Surface(
            x=xx[:, :, 0],
            y=yy[:, :, 0],
            z=np.full_like(xx[:, :, 0], z_min + (z_max - z_min) * 0.5),
            surfacecolor=proba_slice_z,
            colorscale='RdBu',
            showscale=True,
            colorbar=dict(title="Probability"),
            opacity=0.8
        ),
        row=2, col=2
    )
    
    # 3. Class 1概率热图
    proba_class1 = proba[:, 1].reshape(xx.shape)
    proba_slice_z = proba_class1[:, :, grid_size//2]
    fig.add_trace(
        go.Surface(
            x=xx[:, :, 0],
            y=yy[:, :, 0],
            z=np.full_like(xx[:, :, 0], z_min + (z_max - z_min) * 0.5),
            surfacecolor=proba_slice_z,
            colorscale='RdBu_r',
            showscale=True,
            colorbar=dict(title="Probability"),
            opacity=0.8
        ),
        row=2, col=3
    )
    
    # 4. 决策边界（概率=0.5）
    fig.add_trace(
        go.Isosurface(
            x=xx.flatten(),
            y=yy.flatten(),
            z=zz.flatten(),
            value=proba_class0.flatten(),
            isomin=0.5,
            isomax=0.5,
            surface_count=1,
            colorscale=[[0, '#FF6B6B'], [1, '#4ECDC4']],
            opacity=0.6,
            showscale=False
        ),
        row=2, col=4
    )
    
    # 更新第二行场景
    for col in range(1, 5):
        fig.update_scenes(
            dict(
                xaxis_title='Feature 1',
                yaxis_title='Feature 2',
                zaxis_title='Feature 3',
                camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
            ),
            row=2, col=col
        )
    
    # === 第三行：Random Forest概率图 ===
    # 训练Random Forest
    rf = RandomForestClassifier(n_estimators=100)
    rf.fit(X_train, y_train)
    rf_accuracy = rf.score(X_test, y_test)
    
    # 预测概率
    rf_proba = rf.predict_proba(grid_points)
    rf_proba_class0 = rf_proba[:, 0].reshape(xx.shape)
    
    # 1. 概率等值面
    fig.add_trace(
        go.Isosurface(
            x=xx.flatten(),
            y=yy.flatten(),
            z=zz.flatten(),
            value=rf_proba_class0.flatten(),
            isomin=0.3,
            isomax=0.7,
            surface_count=3,
            colorscale='Viridis',
            opacity=0.3,
            showscale=False
        ),
        row=3, col=1
    )
    
    # 添加数据点
    for class_idx in range(n_classes):
        class_mask = (y == class_idx)
        fig.add_trace(
            go.Scatter3d(
                x=X[class_mask, 0],
                y=X[class_mask, 1],
                z=X[class_mask, 2],
                mode='markers',
                marker=dict(
                    size=5,
                    color=colors[class_idx],
                    opacity=0.8
                ),
                showlegend=False
            ),
            row=3, col=1
        )
    
    # 2. Class 0概率热图
    rf_slice_z = rf_proba_class0[:, :, grid_size//2]
    fig.add_trace(
        go.Surface(
            x=xx[:, :, 0],
            y=yy[:, :, 0],
            z=np.full_like(xx[:, :, 0], z_min + (z_max - z_min) * 0.5),
            surfacecolor=rf_slice_z,
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title="Probability"),
            opacity=0.8
        ),
        row=3, col=2
    )
    
    # 3. Class 1概率热图
    rf_proba_class1 = rf_proba[:, 1].reshape(xx.shape)
    rf_slice_z = rf_proba_class1[:, :, grid_size//2]
    fig.add_trace(
        go.Surface(
            x=xx[:, :, 0],
            y=yy[:, :, 0],
            z=np.full_like(xx[:, :, 0], z_min + (z_max - z_min) * 0.5),
            surfacecolor=rf_slice_z,
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title="Probability"),
            opacity=0.8
        ),
        row=3, col=3
    )
    
    # 4. 不确定性图（基于熵）
    epsilon = 1e-10
    entropy = -np.sum(rf_proba * np.log(rf_proba + epsilon), axis=1)
    uncertainty = entropy / np.log(rf_proba.shape[1])
    uncertainty = uncertainty.reshape(xx.shape)
    
    fig.add_trace(
        go.Isosurface(
            x=xx.flatten(),
            y=yy.flatten(),
            z=zz.flatten(),
            value=uncertainty.flatten(),
            isomin=0.5,
            isomax=0.5,
            surface_count=1,
            colorscale='Hot',
            opacity=0.6,
            showscale=True,
            colorbar=dict(title="Uncertainty")
        ),
        row=3, col=4
    )
    
    # 更新第三行场景
    for col in range(1, 5):
        fig.update_scenes(
            dict(
                xaxis_title='Feature 1',
                yaxis_title='Feature 2',
                zaxis_title='Feature 3',
                camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
            ),
            row=3, col=col
        )
    
    # === 第四行：SVM概率图 ===
    # 训练SVM
    svm = SVC(kernel='rbf', probability=True)
    svm.fit(X_train, y_train)
    svm_accuracy = svm.score(X_test, y_test)
    
    # 预测概率
    svm_proba = svm.predict_proba(grid_points)
    svm_proba_class0 = svm_proba[:, 0].reshape(xx.shape)
    
    # 1. 概率等值面
    fig.add_trace(
        go.Isosurface(
            x=xx.flatten(),
            y=yy.flatten(),
            z=zz.flatten(),
            value=svm_proba_class0.flatten(),
            isomin=0.3,
            isomax=0.7,
            surface_count=3,
            colorscale='Blues',
            opacity=0.3,
            showscale=False
        ),
        row=4, col=1
    )
    
    # 添加数据点
    for class_idx in range(n_classes):
        class_mask = (y == class_idx)
        fig.add_trace(
            go.Scatter3d(
                x=X[class_mask, 0],
                y=X[class_mask, 1],
                z=X[class_mask, 2],
                mode='markers',
                marker=dict(
                    size=5,
                    color=colors[class_idx],
                    opacity=0.8
                ),
                showlegend=False
            ),
            row=4, col=1
        )
    
    # 2. Class 0概率热图
    svm_slice_z = svm_proba_class0[:, :, grid_size//2]
    fig.add_trace(
        go.Surface(
            x=xx[:, :, 0],
            y=yy[:, :, 0],
            z=np.full_like(xx[:, :, 0], z_min + (z_max - z_min) * 0.5),
            surfacecolor=svm_slice_z,
            colorscale='Blues',
            showscale=True,
            colorbar=dict(title="Probability"),
            opacity=0.8
        ),
        row=4, col=2
    )
    
    # 3. Class 1概率热图
    svm_proba_class1 = svm_proba[:, 1].reshape(xx.shape)
    svm_slice_z = svm_proba_class1[:, :, grid_size//2]
    fig.add_trace(
        go.Surface(
            x=xx[:, :, 0],
            y=yy[:, :, 0],
            z=np.full_like(xx[:, :, 0], z_min + (z_max - z_min) * 0.5),
            surfacecolor=svm_slice_z,
            colorscale='Blues',
            showscale=True,
            colorbar=dict(title="Probability"),
            opacity=0.8
        ),
        row=4, col=3
    )
    
    # 4. 决策区域
    predictions = svm.predict(grid_points).reshape(xx.shape)
    fig.add_trace(
        go.Volume(
            x=xx.flatten(),
            y=yy.flatten(),
            z=zz.flatten(),
            value=predictions.flatten(),
            isomin=0,
            isomax=1,
            opacity=0.1,
            surface_count=2,
            colorscale=[[0, 'rgba(255,107,107,0.1)'], [1, 'rgba(78,205,196,0.1)']],
            showscale=False
        ),
        row=4, col=4
    )
    
    # 更新第四行场景
    for col in range(1, 5):
        fig.update_scenes(
            dict(
                xaxis_title='Feature 1',
                yaxis_title='Feature 2',
                zaxis_title='Feature 3',
                camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
            ),
            row=4, col=col
        )
    
    # === 第五行：比较视图 ===
    
    # 1. 所有分类器比较（概率热图切片）
    classifiers_proba = []
    for clf_name, clf in classifiers:
        if clf_name == "Logistic Regression":
            clf_proba = proba_class0
        elif clf_name == "Random Forest":
            clf_proba = rf_proba_class0
        elif clf_name == "SVM (RBF)":
            clf_proba = svm_proba_class0
        else:
            clf.fit(X_train, y_train)
            clf_proba = clf.predict_proba(grid_points)[:, 0].reshape(xx.shape)
        
        classifiers_proba.append((clf_name, clf_proba[:, :, grid_size//2]))
    
    # 添加多个曲面进行比较
    z_offset = 0
    for idx, (clf_name, proba_slice) in enumerate(classifiers_proba):
        fig.add_trace(
            go.Surface(
                x=xx[:, :, 0],
                y=yy[:, :, 0],
                z=np.full_like(xx[:, :, 0], z_min + (z_max - z_min) * 0.5 + z_offset),
                surfacecolor=proba_slice,
                colorscale=['Reds', 'Greens', 'Blues', 'Purples', 'Oranges'][idx],
                showscale=False,
                opacity=0.7,
                name=clf_name
            ),
            row=5, col=1
        )
        z_offset += 0.5
    
    # 2. XY平面概率切片
    slice_idx = grid_size // 2
    fig.add_trace(
        go.Surface(
            x=xx[:, :, 0],
            y=yy[:, :, 0],
            z=np.full_like(xx[:, :, 0], z_min + (z_max - z_min) * 0.5),
            surfacecolor=rf_proba_class0[:, :, slice_idx],
            colorscale='RdBu',
            showscale=True,
            colorbar=dict(title="Probability"),
            opacity=0.8
        ),
        row=5, col=2
    )
    
    # 添加数据点
    for class_idx in range(n_classes):
        class_mask = (y == class_idx)
        # 只显示在切片附近的数据点
        z_slice_value = z_min + (z_max - z_min) * 0.5
        mask_near_slice = np.abs(X[class_mask, 2] - z_slice_value) < 1
        if np.any(mask_near_slice):
            fig.add_trace(
                go.Scatter3d(
                    x=X[class_mask, 0][mask_near_slice],
                    y=X[class_mask, 1][mask_near_slice],
                    z=np.full(np.sum(mask_near_slice), z_slice_value),
                    mode='markers',
                    marker=dict(
                        size=8,
                        color=colors[class_idx],
                        opacity=0.8
                    ),
                    showlegend=False
                ),
                row=5, col=2
            )
    
    # 3. XZ平面概率切片
    y_slice_idx = grid_size // 2
    fig.add_trace(
        go.Surface(
            x=xx[:, 0, :],
            y=np.full_like(xx[:, 0, :], y_min + (y_max - y_min) * 0.5),
            z=zz[:, 0, :],
            surfacecolor=rf_proba_class0[:, y_slice_idx, :],
            colorscale='RdBu',
            showscale=False,
            opacity=0.8
        ),
        row=5, col=3
    )
    
    # 4. YZ平面概率切片
    x_slice_idx = grid_size // 2
    fig.add_trace(
        go.Surface(
            x=np.full_like(yy[0, :, :], x_min + (x_max - x_min) * 0.5),
            y=yy[0, :, :],
            z=zz[0, :, :],
            surfacecolor=rf_proba_class0[x_slice_idx, :, :],
            colorscale='RdBu',
            showscale=False,
            opacity=0.8
        ),
        row=5, col=4
    )
    
    # 更新第五行场景
    for col in range(1, 5):
        fig.update_scenes(
            dict(
                xaxis_title='Feature 1',
                yaxis_title='Feature 2',
                zaxis_title='Feature 3',
                camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
            ),
            row=5, col=col
        )
    
    # 更新布局
    fig.update_layout(
        title={
            'text': '3D Probability Map Visualization<br>Interactive Probability Distributions of Different Classifiers',
            'y': 0.98,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': {'size': 24}
        },
        height=2000,
        showlegend=False,
        margin=dict(t=100, b=50, l=50, r=50)
    )
    
    # 保存为单个HTML文件
    print("保存HTML文件...")
    fig.write_html("task3.html", 
                   include_plotlyjs=True,
                   full_html=True,
                   auto_open=False)
    
    print("✅ 单个HTML文件已保存: task3.html")
    
    return fig

if __name__ == "__main__":
    print("="*60)
    print("创建单个HTML文件包含所有3D概率图可视化")
    print("="*60)
    
    fig = create_single_html_with_all_visualizations()
    
    print("\n" + "="*60)
    print("完成!")
    print("文件大小:", f"{len(fig.to_html()) / (1024*1024):.2f} MB")
    print("包含:")
    print("  - 3个3D数据集")
    print("  - 5个分类器的概率图")
    print("  - 多种可视化类型:")
    print("    * 概率等值面")
    print("    * 概率热图")
    print("    * 决策边界")
    print("    * 不确定性图")
    print("    * 概率切片")
    print("="*60)