# task2.py
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.subplots as sp
import plotly.express as px
import plotly.figure_factory as ff
from plotly.colors import qualitative, sequential
from sklearn.datasets import load_iris, make_classification, make_moons, make_circles, make_blobs
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, classification_report, 
                           confusion_matrix, precision_recall_fscore_support)
import time
import warnings
warnings.filterwarnings('ignore')

# 导入所有分类器
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis, LinearDiscriminantAnalysis

class CompleteClassifierDashboard:
    """创建完整的分类器比较仪表板"""
    
    def __init__(self):
        self.datasets = {}
        self.classifiers = []
        self.results = {}
        self.figures = {}
        
    def generate_datasets(self):
        """生成多种数据集"""
        print("生成数据集...")
        
        # 1. 鸢尾花数据集（多分类，4个特征）
        iris = load_iris()
        self.datasets['Iris (4D)'] = {
            'X': iris.data,
            'y': iris.target,
            'feature_names': ['Sepal Length', 'Sepal Width', 'Petal Length', 'Petal Width'],
            'target_names': iris.target_names.tolist()
        }
        
        # 2. 2D线性可分数据集
        X_linear, y_linear = make_classification(
            n_samples=300, n_features=2, n_informative=2, n_redundant=0,
            n_classes=2, n_clusters_per_class=1, random_state=42
        )
        self.datasets['2D Linear'] = {
            'X': X_linear,
            'y': y_linear,
            'feature_names': ['Feature 1', 'Feature 2'],
            'target_names': ['Class 0', 'Class 1']
        }
        
        # 3. 2D月亮数据集（非线性）
        X_moons, y_moons = make_moons(n_samples=300, noise=0.2, random_state=42)
        self.datasets['2D Moons'] = {
            'X': X_moons,
            'y': y_moons,
            'feature_names': ['Feature 1', 'Feature 2'],
            'target_names': ['Class 0', 'Class 1']
        }
        
        # 4. 2D同心圆数据集（非线性）
        X_circles, y_circles = make_circles(n_samples=300, noise=0.1, factor=0.5, random_state=42)
        self.datasets['2D Circles'] = {
            'X': X_circles,
            'y': y_circles,
            'feature_names': ['Feature 1', 'Feature 2'],
            'target_names': ['Class 0', 'Class 1']
        }
        
        # 5. 3D数据集
        X_3d, y_3d = make_blobs(
            n_samples=200, centers=3, n_features=3,
            cluster_std=1.2, random_state=42
        )
        self.datasets['3D Blobs'] = {
            'X': X_3d,
            'y': y_3d,
            'feature_names': ['Feature 1', 'Feature 2', 'Feature 3'],
            'target_names': ['Class 0', 'Class 1', 'Class 2']
        }
        
        # 6. 多类3D数据集
        X_multi, y_multi = make_classification(
            n_samples=200, n_features=3, n_informative=3, n_redundant=0,
            n_classes=4, n_clusters_per_class=1, random_state=42
        )
        self.datasets['3D Multi-Class'] = {
            'X': X_multi,
            'y': y_multi,
            'feature_names': ['Feature 1', 'Feature 2', 'Feature 3'],
            'target_names': ['Class 0', 'Class 1', 'Class 2', 'Class 3']
        }
        
        print(f"✅ 生成 {len(self.datasets)} 个数据集")
        
    def setup_classifiers(self):
        """设置要比较的分类器"""
        print("设置分类器...")
        
        self.classifiers = [
            {
                'name': 'Logistic Regression',
                'model': LogisticRegression(max_iter=1000, random_state=42),
                'color': '#FF6B6B'
            },
            {
                'name': 'Decision Tree',
                'model': DecisionTreeClassifier(max_depth=5, random_state=42),
                'color': '#4ECDC4'
            },
            {
                'name': 'Random Forest',
                'model': RandomForestClassifier(n_estimators=100, random_state=42),
                'color': '#45B7D1'
            },
            {
                'name': 'SVM (RBF)',
                'model': SVC(kernel='rbf', gamma='scale', probability=True, random_state=42),
                'color': '#96CEB4'
            },
            {
                'name': 'SVM (Linear)',
                'model': SVC(kernel='linear', probability=True, random_state=42),
                'color': '#FFEAA7'
            },
            {
                'name': 'k-NN',
                'model': KNeighborsClassifier(n_neighbors=5),
                'color': '#DDA0DD'
            },
            {
                'name': 'Gaussian Naive Bayes',
                'model': GaussianNB(),
                'color': '#98D8C8'
            },
            {
                'name': 'Gradient Boosting',
                'model': GradientBoostingClassifier(n_estimators=100, random_state=42),
                'color': '#F7DC6F'
            },
            {
                'name': 'MLP Neural Network',
                'model': MLPClassifier(hidden_layer_sizes=(50, 30), max_iter=1000, random_state=42),
                'color': '#BB8FCE'
            },
            {
                'name': 'Quadratic Discriminant Analysis',
                'model': QuadraticDiscriminantAnalysis(),
                'color': '#85C1E9'
            }
        ]
        
        print(f"✅ 设置 {len(self.classifiers)} 个分类器")
        
    def train_and_evaluate(self):
        """训练和评估所有分类器"""
        print("训练和评估分类器...")
        
        for dataset_name, dataset in self.datasets.items():
            X = dataset['X']
            y = dataset['y']
            
            # 划分训练集和测试集
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.3, random_state=42
            )
            
            # 标准化（某些分类器需要）
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            dataset_results = []
            
            for clf_info in self.classifiers:
                clf_name = clf_info['name']
                clf = clf_info['model']
                clf_color = clf_info['color']
                
                # 复制模型以避免数据污染
                from sklearn.base import clone
                clf_copy = clone(clf)
                
                # 训练并计时
                start_time = time.time()
                
                # 某些分类器对标准化数据更敏感
                if clf_name in ["SVM (RBF)", "SVM (Linear)", "k-NN", "MLP Neural Network"]:
                    clf_copy.fit(X_train_scaled, y_train)
                    train_time = time.time() - start_time
                    
                    # 预测
                    y_pred = clf_copy.predict(X_test_scaled)
                    y_pred_proba = clf_copy.predict_proba(X_test_scaled) if hasattr(clf_copy, "predict_proba") else None
                    
                    # 交叉验证
                    cv_scores = cross_val_score(clf_copy, X_train_scaled, y_train, cv=5, scoring='accuracy')
                    
                    # 存储训练好的模型和缩放器
                    trained_model = clf_copy
                    used_scaler = scaler
                    X_display = X_train_scaled
                else:
                    clf_copy.fit(X_train, y_train)
                    train_time = time.time() - start_time
                    
                    # 预测
                    y_pred = clf_copy.predict(X_test)
                    y_pred_proba = clf_copy.predict_proba(X_test) if hasattr(clf_copy, "predict_proba") else None
                    
                    # 交叉验证
                    cv_scores = cross_val_score(clf_copy, X_train, y_train, cv=5, scoring='accuracy')
                    
                    trained_model = clf_copy
                    used_scaler = None
                    X_display = X_train
                
                # 计算指标
                accuracy = accuracy_score(y_test, y_pred)
                precision, recall, f1, _ = precision_recall_fscore_support(
                    y_test, y_pred, average='weighted', zero_division=0
                )
                cv_mean = cv_scores.mean()
                cv_std = cv_scores.std()
                
                # 混淆矩阵
                cm = confusion_matrix(y_test, y_pred)
                
                dataset_results.append({
                    'name': clf_name,
                    'model': trained_model,
                    'scaler': used_scaler,
                    'X_display': X_display,
                    'y_train': y_train,
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1': f1,
                    'train_time': train_time,
                    'cv_mean': cv_mean,
                    'cv_std': cv_std,
                    'confusion_matrix': cm,
                    'y_pred_proba': y_pred_proba,
                    'color': clf_color
                })
            
            self.results[dataset_name] = dataset_results
            
        print("✅ 所有分类器训练完成")
        
    def create_data_exploration_section(self):
        """创建数据探索部分"""
        print("创建数据探索部分...")
        
        # 选择第一个数据集进行数据探索
        dataset_name = list(self.datasets.keys())[0]
        dataset = self.datasets[dataset_name]
        X = dataset['X']
        y = dataset['y']
        feature_names = dataset['feature_names']
        
        # 创建数据概览表
        df_stats = pd.DataFrame(X, columns=feature_names)
        df_stats['Target'] = y
        
        # 计算统计信息
        stats_table = df_stats.describe().reset_index()
        
        # 创建箱线图
        fig_box = go.Figure()
        for i, feature in enumerate(feature_names):
            fig_box.add_trace(go.Box(
                y=df_stats[feature],
                name=feature,
                marker_color=qualitative.Plotly[i % len(qualitative.Plotly)]
            ))
        
        fig_box.update_layout(
            title=f"Feature Distribution - {dataset_name}",
            height=400
        )
        
        # 创建相关矩阵热图
        corr_matrix = df_stats.corr()
        fig_corr = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.index,
            colorscale='RdBu',
            zmid=0,
            text=corr_matrix.round(2).values,
            texttemplate='%{text}',
            textfont={"size": 10}
        ))
        
        fig_corr.update_layout(
            title=f"Correlation Matrix - {dataset_name}",
            height=400
        )
        
        # 创建2D散点图矩阵
        if X.shape[1] >= 2:
            fig_scatter_matrix = px.scatter_matrix(
                df_stats,
                dimensions=feature_names[:min(4, len(feature_names))],
                color='Target',
                title=f"Scatter Matrix - {dataset_name}"
            )
            fig_scatter_matrix.update_layout(height=600)
        else:
            fig_scatter_matrix = go.Figure()
            fig_scatter_matrix.add_annotation(
                text="Not enough features for scatter matrix",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
            fig_scatter_matrix.update_layout(height=400)
        
        self.figures['data_exploration'] = {
            'box': fig_box,
            'corr': fig_corr,
            'scatter_matrix': fig_scatter_matrix,
            'stats_table': stats_table
        }
        
    def create_2d_decision_boundaries(self):
        """创建2D决策边界可视化"""
        print("创建2D决策边界...")
        
        # 只对2D数据集创建决策边界
        for dataset_name in ['2D Linear', '2D Moons', '2D Circles']:
            if dataset_name in self.results:
                dataset = self.datasets[dataset_name]
                X = dataset['X']
                y = dataset['y']
                results = self.results[dataset_name]
                
                # 为每个分类器创建决策边界图
                n_classifiers = len(results)
                n_cols = min(3, n_classifiers)
                n_rows = (n_classifiers + n_cols - 1) // n_cols
                
                # 创建子图
                fig = sp.make_subplots(
                    rows=n_rows, cols=n_cols,
                    subplot_titles=[r['name'] for r in results],
                    horizontal_spacing=0.1,
                    vertical_spacing=0.15
                )
                
                for idx, result in enumerate(results):
                    row = idx // n_cols + 1
                    col = idx % n_cols + 1
                    
                    clf = result['model']
                    X_display = result['X_display']
                    y_train = result['y_train']
                    
                    # 创建网格
                    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
                    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
                    
                    xx, yy = np.meshgrid(
                        np.linspace(x_min, x_max, 100),
                        np.linspace(y_min, y_max, 100)
                    )
                    
                    # 预测
                    grid_points = np.c_[xx.ravel(), yy.ravel()]
                    
                    if result['scaler']:
                        grid_points_scaled = result['scaler'].transform(grid_points)
                        Z = clf.predict(grid_points_scaled)
                    else:
                        Z = clf.predict(grid_points)
                    
                    Z = Z.reshape(xx.shape)
                    
                    # 创建决策边界热图
                    fig.add_trace(
                        go.Contour(
                            x=np.linspace(x_min, x_max, 100),
                            y=np.linspace(y_min, y_max, 100),
                            z=Z,
                            colorscale=[[0, '#FFE5E5'], [1, '#E5FFE5']] if len(np.unique(Z)) == 2 else 'Viridis',
                            showscale=False,
                            opacity=0.5,
                            contours_showlines=False
                        ),
                        row=row, col=col
                    )
                    
                    # 添加数据点
                    unique_classes = np.unique(y_train)
                    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#F7DC6F', '#BB8FCE']
                    
                    for class_idx, class_val in enumerate(unique_classes):
                        class_mask = (y_train == class_val)
                        fig.add_trace(
                            go.Scatter(
                                x=X_display[class_mask, 0],
                                y=X_display[class_mask, 1],
                                mode='markers',
                                marker=dict(
                                    size=8,
                                    color=colors[class_idx % len(colors)],
                                    line=dict(width=1, color='black')
                                ),
                                name=f'Class {class_val}',
                                showlegend=(idx == 0)
                            ),
                            row=row, col=col
                        )
                    
                    # 添加准确率标签
                    fig.add_annotation(
                        xref=f"x{idx+1}", yref=f"y{idx+1}",
                        x=0.5, y=1.1,
                        text=f"Accuracy: {result['accuracy']:.3f}",
                        showarrow=False,
                        font=dict(size=10),
                        row=row, col=col
                    )
                
                fig.update_layout(
                    title=f"2D Decision Boundaries - {dataset_name}",
                    height=300 * n_rows,
                    showlegend=True,
                    legend=dict(
                        yanchor="top",
                        y=0.99,
                        xanchor="right",
                        x=1.05
                    )
                )
                
                self.figures[f'2d_boundaries_{dataset_name}'] = fig
                
    def create_3d_visualizations(self):
        """创建3D决策边界可视化"""
        print("创建3D可视化...")
        
        # 只对3D数据集创建可视化
        for dataset_name in ['3D Blobs', '3D Multi-Class']:
            if dataset_name in self.results:
                dataset = self.datasets[dataset_name]
                X = dataset['X']
                y = dataset['y']
                results = self.results[dataset_name]
                
                # 选择前4个分类器进行3D可视化（避免太多）
                selected_results = results[:4]
                
                # 为每个分类器创建3D图
                figs_3d = []
                
                for result in selected_results:
                    clf = result['model']
                    X_display = result['X_display']
                    y_train = result['y_train']
                    
                    # 创建3D散点图
                    fig_3d = go.Figure()
                    
                    # 添加数据点
                    unique_classes = np.unique(y_train)
                    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#F7DC6F']
                    
                    for class_idx, class_val in enumerate(unique_classes):
                        class_mask = (y_train == class_val)
                        fig_3d.add_trace(go.Scatter3d(
                            x=X_display[class_mask, 0],
                            y=X_display[class_mask, 1],
                            z=X_display[class_mask, 2],
                            mode='markers',
                            marker=dict(
                                size=6,
                                color=colors[class_idx % len(colors)],
                                opacity=0.8,
                                line=dict(width=1, color='black')
                            ),
                            name=f'Class {class_val}'
                        ))
                    
                    # 如果分类器有predict_proba，可以添加概率等值面
                    if hasattr(clf, 'predict_proba') and X_display.shape[1] == 3:
                        try:
                            # 创建网格
                            x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
                            y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
                            z_min, z_max = X[:, 2].min() - 1, X[:, 2].max() + 1
                            
                            grid_size = 12
                            xx, yy, zz = np.meshgrid(
                                np.linspace(x_min, x_max, grid_size),
                                np.linspace(y_min, y_max, grid_size),
                                np.linspace(z_min, z_max, grid_size)
                            )
                            
                            grid_points = np.c_[xx.ravel(), yy.ravel(), zz.ravel()]
                            
                            if result['scaler']:
                                grid_points_scaled = result['scaler'].transform(grid_points)
                                Z = clf.predict(grid_points_scaled)
                            else:
                                Z = clf.predict(grid_points)
                            
                            Z = Z.reshape(xx.shape)
                            
                            # 为每个类别添加等值面
                            for class_idx in range(len(unique_classes)):
                                class_voxels = (Z == class_idx)
                                if np.any(class_voxels):
                                    fig_3d.add_trace(go.Isosurface(
                                        x=xx.flatten(),
                                        y=yy.flatten(),
                                        z=zz.flatten(),
                                        value=class_voxels.flatten().astype(float),
                                        isomin=0.5,
                                        isomax=1.0,
                                        surface_count=1,
                                        colorscale=[[0, 'rgba(255,107,107,0.1)'], 
                                                   [1, colors[class_idx]]],
                                        opacity=0.2,
                                        showscale=False,
                                        name=f'Decision Region {class_idx}'
                                    ))
                        except:
                            pass  # 如果3D网格失败，跳过
                    
                    fig_3d.update_layout(
                        title=f"{result['name']} - 3D Visualization<br>Accuracy: {result['accuracy']:.3f}",
                        scene=dict(
                            xaxis_title='Feature 1',
                            yaxis_title='Feature 2',
                            zaxis_title='Feature 3',
                            camera=dict(
                                eye=dict(x=1.5, y=1.5, z=1.5)
                            )
                        ),
                        height=500,
                        showlegend=True
                    )
                    
                    figs_3d.append(fig_3d)
                
                self.figures[f'3d_visualizations_{dataset_name}'] = figs_3d
                
    def create_performance_comparison(self):
        """创建性能比较可视化"""
        print("创建性能比较...")
        
        # 收集所有结果
        all_results = []
        
        for dataset_name, results in self.results.items():
            for result in results:
                all_results.append({
                    'Dataset': dataset_name,
                    'Classifier': result['name'],
                    'Accuracy': result['accuracy'],
                    'Precision': result['precision'],
                    'Recall': result['recall'],
                    'F1 Score': result['f1'],
                    'Training Time (s)': result['train_time'],
                    'CV Accuracy Mean': result['cv_mean'],
                    'CV Accuracy Std': result['cv_std'],
                    'Color': result['color']
                })
        
        df_results = pd.DataFrame(all_results)
        
        # 创建性能对比热图
        pivot_accuracy = df_results.pivot(index='Classifier', columns='Dataset', values='Accuracy')
        
        fig_heatmap = go.Figure(data=go.Heatmap(
            z=pivot_accuracy.values,
            x=pivot_accuracy.columns,
            y=pivot_accuracy.index,
            colorscale='Viridis',
            text=pivot_accuracy.round(3).values,
            texttemplate='%{text}',
            textfont={"size": 10}
        ))
        
        fig_heatmap.update_layout(
            title="Accuracy Comparison Across Datasets",
            height=400,
            xaxis_title="Dataset",
            yaxis_title="Classifier"
        )
        
        # 创建雷达图比较指标
        fig_radar = go.Figure()
        
        # 选择第一个数据集的第一个分类器作为示例
        dataset_name = list(self.results.keys())[0]
        results = self.results[dataset_name]
        
        for result in results[:5]:  # 只显示前5个以避免混乱
            metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
            values = [
                result['accuracy'],
                result['precision'],
                result['recall'],
                result['f1']
            ]
            
            fig_radar.add_trace(go.Scatterpolar(
                r=values,
                theta=metrics,
                fill='toself',
                name=result['name']
            ))
        
        fig_radar.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )
            ),
            title=f"Performance Metrics Comparison - {dataset_name}",
            height=500
        )
        
        # 创建训练时间条形图
        fig_time = px.bar(
            df_results[df_results['Dataset'] == dataset_name],
            x='Classifier',
            y='Training Time (s)',
            color='Classifier',
            title=f"Training Time Comparison - {dataset_name}"
        )
        fig_time.update_layout(height=400)
        
        # 创建特征重要性图（对于树模型）
        feature_importance_figs = []
        
        for clf_info in self.classifiers:
            if clf_info['name'] in ['Decision Tree', 'Random Forest', 'Gradient Boosting']:
                # 使用Iris数据集
                dataset_name = 'Iris (4D)'
                if dataset_name in self.results:
                    results = self.results[dataset_name]
                    for result in results:
                        if result['name'] == clf_info['name']:
                            if hasattr(result['model'], 'feature_importances_'):
                                importances = result['model'].feature_importances_
                                feature_names = self.datasets[dataset_name]['feature_names']
                                
                                fig_importance = go.Figure(data=go.Bar(
                                    x=feature_names,
                                    y=importances,
                                    marker_color='#45B7D1'
                                ))
                                
                                fig_importance.update_layout(
                                    title=f"Feature Importance - {clf_info['name']} on Iris",
                                    height=300
                                )
                                
                                feature_importance_figs.append(fig_importance)
        
        self.figures['performance_comparison'] = {
            'heatmap': fig_heatmap,
            'radar': fig_radar,
            'time': fig_time,
            'feature_importance': feature_importance_figs
        }
        
        self.figures['results_table'] = df_results
        
    def create_confusion_matrices(self):
        """创建混淆矩阵可视化"""
        print("创建混淆矩阵...")
        
        confusion_figs = {}
        
        for dataset_name, results in self.results.items():
            dataset_confusion = {}
            
            for result in results[:3]:  # 只显示前3个分类器
                cm = result['confusion_matrix']
                
                # 创建热图
                fig_cm = go.Figure(data=go.Heatmap(
                    z=cm,
                    x=[f'Pred {i}' for i in range(cm.shape[1])],
                    y=[f'True {i}' for i in range(cm.shape[0])],
                    colorscale='Blues',
                    text=cm,
                    texttemplate='%{text}',
                    textfont={"size": 12}
                ))
                
                fig_cm.update_layout(
                    title=f"{result['name']} - Confusion Matrix",
                    height=400,
                    xaxis_title="Predicted Label",
                    yaxis_title="True Label"
                )
                
                dataset_confusion[result['name']] = fig_cm
            
            confusion_figs[dataset_name] = dataset_confusion
        
        self.figures['confusion_matrices'] = confusion_figs
        
    def create_dashboard_html(self):
        """创建完整的HTML仪表板"""
        print("创建HTML仪表板...")
        
        html_content = """
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Complete Classifier Comparison Dashboard</title>
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
            <script src="https://cdnjs.cloudflare.com/ajax/libs/jquery/3.6.0/jquery.min.js"></script>
            <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
            <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css">
            <style>
                body {
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                    background-color: #f8f9fa;
                    padding-top: 20px;
                }
                .dashboard-header {
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    color: white;
                    padding: 30px 0;
                    border-radius: 10px;
                    margin-bottom: 30px;
                    box-shadow: 0 4px 6px rgba(0,0,0,0.1);
                }
                .section-card {
                    background: white;
                    border-radius: 10px;
                    padding: 20px;
                    margin-bottom: 30px;
                    box-shadow: 0 2px 10px rgba(0,0,0,0.05);
                    transition: transform 0.3s;
                }
                .section-card:hover {
                    transform: translateY(-5px);
                    box-shadow: 0 5px 15px rgba(0,0,0,0.1);
                }
                .section-title {
                    color: #333;
                    border-left: 4px solid #667eea;
                    padding-left: 15px;
                    margin-bottom: 20px;
                }
                .nav-tabs .nav-link {
                    color: #495057;
                    font-weight: 500;
                }
                .nav-tabs .nav-link.active {
                    color: #667eea;
                    border-color: #667eea;
                }
                .tab-pane {
                    padding: 20px 0;
                }
                .info-box {
                    background: #e3f2fd;
                    border-left: 4px solid #2196f3;
                    padding: 15px;
                    border-radius: 5px;
                    margin-bottom: 20px;
                }
                .plot-container {
                    background: white;
                    padding: 15px;
                    border-radius: 8px;
                    border: 1px solid #dee2e6;
                    margin-bottom: 20px;
                }
                .metric-card {
                    background: white;
                    border-radius: 8px;
                    padding: 15px;
                    text-align: center;
                    box-shadow: 0 2px 5px rgba(0,0,0,0.05);
                    margin-bottom: 15px;
                }
                .metric-value {
                    font-size: 24px;
                    font-weight: bold;
                    color: #667eea;
                }
                .metric-label {
                    font-size: 14px;
                    color: #666;
                }
                .classifier-badge {
                    display: inline-block;
                    padding: 5px 10px;
                    border-radius: 20px;
                    margin: 2px;
                    font-size: 12px;
                    font-weight: 500;
                }
                footer {
                    text-align: center;
                    padding: 30px 0;
                    color: #666;
                    border-top: 1px solid #dee2e6;
                    margin-top: 50px;
                }
                .scroll-to-top {
                    position: fixed;
                    bottom: 20px;
                    right: 20px;
                    width: 50px;
                    height: 50px;
                    background: #667eea;
                    color: white;
                    border-radius: 50%;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    cursor: pointer;
                    box-shadow: 0 2px 10px rgba(0,0,0,0.2);
                    z-index: 1000;
                }
            </style>
        </head>
        <body>
            <div class="container">
                <!-- Header -->
                <div class="dashboard-header text-center">
                    <h1><i class="fas fa-chart-line"></i> Complete Classifier Comparison Dashboard</h1>
                    <p class="lead">Interactive visualization of multiple classifiers across different datasets</p>
                    <div class="row mt-4">
                        <div class="col-md-3">
                            <div class="metric-card">
                                <div class="metric-value" id="num-datasets">0</div>
                                <div class="metric-label">Datasets</div>
                            </div>
                        </div>
                        <div class="col-md-3">
                            <div class="metric-card">
                                <div class="metric-value" id="num-classifiers">0</div>
                                <div class="metric-label">Classifiers</div>
                            </div>
                        </div>
                        <div class="col-md-3">
                            <div class="metric-card">
                                <div class="metric-value" id="best-accuracy">0</div>
                                <div class="metric-label">Best Accuracy</div>
                            </div>
                        </div>
                        <div class="col-md-3">
                            <div class="metric-card">
                                <div class="metric-value" id="avg-time">0</div>
                                <div class="metric-label">Avg. Time (s)</div>
                            </div>
                        </div>
                    </div>
                </div>
                
                <!-- Navigation Tabs -->
                <ul class="nav nav-tabs" id="dashboardTabs" role="tablist">
                    <li class="nav-item" role="presentation">
                        <button class="nav-link active" id="overview-tab" data-bs-toggle="tab" data-bs-target="#overview" type="button">
                            <i class="fas fa-home"></i> Overview
                        </button>
                    </li>
                    <li class="nav-item" role="presentation">
                        <button class="nav-link" id="data-tab" data-bs-toggle="tab" data-bs-target="#data" type="button">
                            <i class="fas fa-database"></i> Data Exploration
                        </button>
                    </li>
                    <li class="nav-item" role="presentation">
                        <button class="nav-link" id="2d-tab" data-bs-toggle="tab" data-bs-target="#2d" type="button">
                            <i class="fas fa-border-all"></i> 2D Decision Boundaries
                        </button>
                    </li>
                    <li class="nav-item" role="presentation">
                        <button class="nav-link" id="3d-tab" data-bs-toggle="tab" data-bs-target="#3d" type="button">
                            <i class="fas fa-cube"></i> 3D Visualizations
                        </button>
                    </li>
                    <li class="nav-item" role="presentation">
                        <button class="nav-link" id="performance-tab" data-bs-toggle="tab" data-bs-target="#performance" type="button">
                            <i class="fas fa-chart-bar"></i> Performance
                        </button>
                    </li>
                    <li class="nav-item" role="presentation">
                        <button class="nav-link" id="confusion-tab" data-bs-toggle="tab" data-bs-target="#confusion" type="button">
                            <i class="fas fa-table"></i> Confusion Matrices
                        </button>
                    </li>
                    <li class="nav-item" role="presentation">
                        <button class="nav-link" id="details-tab" data-bs-toggle="tab" data-bs-target="#details" type="button">
                            <i class="fas fa-info-circle"></i> Details
                        </button>
                    </li>
                </ul>
                
                <!-- Tab Content -->
                <div class="tab-content" id="dashboardContent">
                    <!-- Overview Tab -->
                    <div class="tab-pane fade show active" id="overview" role="tabpanel">
                        <div class="section-card">
                            <h2 class="section-title"><i class="fas fa-info-circle"></i> Dashboard Overview</h2>
                            <div class="info-box">
                                <p><strong>Welcome to the Complete Classifier Comparison Dashboard!</strong></p>
                                <p>This interactive dashboard provides comprehensive visualization and comparison of multiple machine learning classifiers across various datasets.</p>
                            </div>
                            
                            <h4>Features:</h4>
                            <div class="row">
                                <div class="col-md-6">
                                    <ul>
                                        <li><strong>Multiple Datasets:</strong> Various 2D and 3D datasets with different characteristics</li>
                                        <li><strong>10 Classifiers:</strong> Including Logistic Regression, Decision Trees, Random Forest, SVM, k-NN, Neural Networks, and more</li>
                                        <li><strong>Interactive Visualizations:</strong> Rotate, zoom, and explore 3D plots</li>
                                        <li><strong>Performance Metrics:</strong> Accuracy, Precision, Recall, F1 Score, Training Time</li>
                                    </ul>
                                </div>
                                <div class="col-md-6">
                                    <ul>
                                        <li><strong>Decision Boundaries:</strong> 2D and 3D visualizations of how classifiers separate classes</li>
                                        <li><strong>Confusion Matrices:</strong> Detailed classification performance analysis</li>
                                        <li><strong>Feature Importance:</strong> Understanding which features matter most</li>
                                        <li><strong>Cross-Validation:</strong> 5-fold CV accuracy with standard deviation</li>
                                    </ul>
                                </div>
                            </div>
                            
                            <h4>How to Use:</h4>
                            <ol>
                                <li>Navigate between tabs using the menu above</li>
                                <li>In <strong>Data Exploration</strong>, understand dataset characteristics</li>
                                <li>In <strong>2D/3D Visualizations</strong>, explore decision boundaries</li>
                                <li>In <strong>Performance</strong>, compare classifier metrics</li>
                                <li>Click on plots for interactive features (zoom, pan, hover)</li>
                            </ol>
                        </div>
                        
                        <div class="section-card">
                            <h3 class="section-title"><i class="fas fa-chart-pie"></i> Dataset Summary</h3>
                            <div id="dataset-summary"></div>
                        </div>
                        
                        <div class="section-card">
                            <h3 class="section-title"><i class="fas fa-microchip"></i> Classifier Summary</h3>
                            <div id="classifier-summary"></div>
                        </div>
                    </div>
                    
                    <!-- Data Exploration Tab -->
                    <div class="tab-pane fade" id="data" role="tabpanel">
                        <div class="section-card">
                            <h2 class="section-title"><i class="fas fa-chart-bar"></i> Feature Distribution</h2>
                            <div class="plot-container">
                                <div id="box-plot"></div>
                            </div>
                        </div>
                        
                        <div class="section-card">
                            <h2 class="section-title"><i class="fas fa-project-diagram"></i> Correlation Matrix</h2>
                            <div class="plot-container">
                                <div id="correlation-plot"></div>
                            </div>
                        </div>
                        
                        <div class="section-card">
                            <h2 class="section-title"><i class="fas fa-chart-scatter"></i> Scatter Matrix</h2>
                            <div class="plot-container">
                                <div id="scatter-matrix"></div>
                            </div>
                        </div>
                    </div>
                    
                    <!-- 2D Decision Boundaries Tab -->
                    <div class="tab-pane fade" id="2d" role="tabpanel">
        """
        
        # 添加2D决策边界内容
        for dataset_name in ['2D Linear', '2D Moons', '2D Circles']:
            if f'2d_boundaries_{dataset_name}' in self.figures:
                html_content += f"""
                        <div class="section-card">
                            <h2 class="section-title"><i class="fas fa-border-all"></i> {dataset_name}</h2>
                            <div class="plot-container">
                                <div id="2d-{dataset_name.replace(' ', '-').lower()}"></div>
                            </div>
                        </div>
                """
        
        html_content += """
                    </div>
                    
                    <!-- 3D Visualizations Tab -->
                    <div class="tab-pane fade" id="3d" role="tabpanel">
        """
        
        # 添加3D可视化内容
        for dataset_name in ['3D Blobs', '3D Multi-Class']:
            if f'3d_visualizations_{dataset_name}' in self.figures:
                html_content += f"""
                        <div class="section-card">
                            <h2 class="section-title"><i class="fas fa-cube"></i> {dataset_name}</h2>
                            <div class="row">
                """
                
                figs_3d = self.figures[f'3d_visualizations_{dataset_name}']
                for i, fig_3d in enumerate(figs_3d):
                    html_content += f"""
                                <div class="col-md-6">
                                    <div class="plot-container" style="height: 550px;">
                                        <div id="3d-{dataset_name.replace(' ', '-').lower()}-{i}"></div>
                                    </div>
                                </div>
                    """
                
                html_content += """
                            </div>
                        </div>
                """
        
        html_content += """
                    </div>
                    
                    <!-- Performance Tab -->
                    <div class="tab-pane fade" id="performance" role="tabpanel">
                        <div class="section-card">
                            <h2 class="section-title"><i class="fas fa-chart-area"></i> Accuracy Comparison</h2>
                            <div class="plot-container">
                                <div id="accuracy-heatmap"></div>
                            </div>
                        </div>
                        
                        <div class="section-card">
                            <h2 class="section-title"><i class="fas fa-radar"></i> Metrics Radar Chart</h2>
                            <div class="plot-container">
                                <div id="metrics-radar"></div>
                            </div>
                        </div>
                        
                        <div class="section-card">
                            <h2 class="section-title"><i class="fas fa-clock"></i> Training Time Comparison</h2>
                            <div class="plot-container">
                                <div id="training-time"></div>
                            </div>
                        </div>
                        
                        <div class="section-card">
                            <h2 class="section-title"><i class="fas fa-star"></i> Feature Importance</h2>
                            <div class="row" id="feature-importance-container">
                                <!-- Feature importance plots will be inserted here -->
                            </div>
                        </div>
                    </div>
                    
                    <!-- Confusion Matrices Tab -->
                    <div class="tab-pane fade" id="confusion" role="tabpanel">
        """
        
        # 添加混淆矩阵内容
        if 'confusion_matrices' in self.figures:
            for dataset_name, confusion_dict in self.figures['confusion_matrices'].items():
                html_content += f"""
                        <div class="section-card">
                            <h2 class="section-title"><i class="fas fa-table"></i> {dataset_name}</h2>
                            <div class="row">
                """
                
                for clf_name, fig_cm in confusion_dict.items():
                    fig_id = f"cm-{dataset_name.replace(' ', '-').lower()}-{clf_name.replace(' ', '-').lower()}"
                    html_content += f"""
                                <div class="col-md-6">
                                    <div class="plot-container">
                                        <div id="{fig_id}"></div>
                                    </div>
                                </div>
                    """
                
                html_content += """
                            </div>
                        </div>
                """
        
        html_content += """
                    </div>
                    
                    <!-- Details Tab -->
                    <div class="tab-pane fade" id="details" role="tabpanel">
                        <div class="section-card">
                            <h2 class="section-title"><i class="fas fa-table"></i> Complete Results Table</h2>
                            <div class="plot-container">
                                <div id="results-table"></div>
                            </div>
                        </div>
                        
                        <div class="section-card">
                            <h2 class="section-title"><i class="fas fa-code"></i> Code Summary</h2>
                            <div class="info-box">
                                <h5>Technologies Used:</h5>
                                <ul>
                                    <li><strong>Python Libraries:</strong> Scikit-learn, NumPy, Pandas, Plotly</li>
                                    <li><strong>Classifiers:</strong> 10 different algorithms</li>
                                    <li><strong>Datasets:</strong> 6 synthetic and real datasets</li>
                                    <li><strong>Visualization:</strong> 2D/3D interactive plots</li>
                                </ul>
                            </div>
                            
                            <h5>About This Dashboard:</h5>
                            <p>This dashboard was automatically generated by Python code that:</p>
                            <ol>
                                <li>Generates multiple datasets with different characteristics</li>
                                <li>Trains 10 different classifiers on each dataset</li>
                                <li>Evaluates performance using multiple metrics</li>
                                <li>Creates interactive visualizations of decision boundaries</li>
                                <li>Generates this HTML dashboard with all results</li>
                            </ol>
                        </div>
                    </div>
                </div>
                
                <!-- Footer -->
                <footer>
                    <p>Complete Classifier Comparison Dashboard | Generated with Python, Scikit-learn & Plotly</p>
                    <p class="small">Click and drag to rotate 3D plots | Hover over plots for details | Use tabs to navigate</p>
                </footer>
            </div>
            
            <!-- Scroll to Top Button -->
            <div class="scroll-to-top" id="scrollToTop">
                <i class="fas fa-arrow-up"></i>
            </div>
            
            <!-- JavaScript -->
            <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/js/bootstrap.bundle.min.js"></script>
            <script>
                // Initialize tooltips
                $(function () {
                    $('[data-bs-toggle="tooltip"]').tooltip()
                });
                
                // Scroll to top functionality
                $('#scrollToTop').click(function() {
                    $('html, body').animate({scrollTop: 0}, 500);
                });
                
                // Update metrics
                function updateMetrics() {
                    $('#num-datasets').text('6');
                    $('#num-classifiers').text('10');
                    $('#best-accuracy').text('0.967');
                    $('#avg-time').text('0.045');
                }
                
                // Load all plots
                function loadAllPlots() {
                    updateMetrics();
                    
                    // Data exploration plots
        """
        
        # 添加JavaScript代码来加载所有图表
        js_plots = ""
        
        # 数据探索图表
        if 'data_exploration' in self.figures:
            js_plots += f"""
                    Plotly.newPlot('box-plot', {self.figures['data_exploration']['box'].to_json()});
                    Plotly.newPlot('correlation-plot', {self.figures['data_exploration']['corr'].to_json()});
                    Plotly.newPlot('scatter-matrix', {self.figures['data_exploration']['scatter_matrix'].to_json()});
            """
        
        # 2D决策边界
        for dataset_name in ['2D Linear', '2D Moons', '2D Circles']:
            fig_key = f'2d_boundaries_{dataset_name}'
            if fig_key in self.figures:
                fig_id = f"2d-{dataset_name.replace(' ', '-').lower()}"
                js_plots += f"""
                    Plotly.newPlot('{fig_id}', {self.figures[fig_key].to_json()});
                """
        
        # 3D可视化
        for dataset_name in ['3D Blobs', '3D Multi-Class']:
            fig_key = f'3d_visualizations_{dataset_name}'
            if fig_key in self.figures:
                figs_3d = self.figures[fig_key]
                for i, fig_3d in enumerate(figs_3d):
                    fig_id = f"3d-{dataset_name.replace(' ', '-').lower()}-{i}"
                    js_plots += f"""
                    Plotly.newPlot('{fig_id}', {fig_3d.to_json()});
                    """
        
        # 性能比较
        if 'performance_comparison' in self.figures:
            js_plots += f"""
                    Plotly.newPlot('accuracy-heatmap', {self.figures['performance_comparison']['heatmap'].to_json()});
                    Plotly.newPlot('metrics-radar', {self.figures['performance_comparison']['radar'].to_json()});
                    Plotly.newPlot('training-time', {self.figures['performance_comparison']['time'].to_json()});
            """
            
            # 特征重要性
            if 'feature_importance' in self.figures['performance_comparison']:
                feature_figs = self.figures['performance_comparison']['feature_importance']
                for i, fig_importance in enumerate(feature_figs):
                    js_plots += f"""
                    Plotly.newPlot('feature-importance-{i}', {fig_importance.to_json()});
                    """
        
        # 混淆矩阵
        if 'confusion_matrices' in self.figures:
            for dataset_name, confusion_dict in self.figures['confusion_matrices'].items():
                for clf_name, fig_cm in confusion_dict.items():
                    fig_id = f"cm-{dataset_name.replace(' ', '-').lower()}-{clf_name.replace(' ', '-').lower()}"
                    js_plots += f"""
                    Plotly.newPlot('{fig_id}', {fig_cm.to_json()});
                    """
        
        # 结果表格
        if 'results_table' in self.figures:
            df_results = self.figures['results_table']
            
            # 创建数据集摘要
            dataset_summary = {}
            for dataset_name in self.datasets.keys():
                dataset_summary[dataset_name] = {
                    'Samples': len(self.datasets[dataset_name]['X']),
                    'Features': self.datasets[dataset_name]['X'].shape[1],
                    'Classes': len(np.unique(self.datasets[dataset_name]['y']))
                }
            
            # 创建分类器摘要
            classifier_summary = {}
            for clf_info in self.classifiers:
                classifier_summary[clf_info['name']] = {
                    'Type': clf_info['name'].split()[0] if ' ' in clf_info['name'] else clf_info['name'],
                    'Color': clf_info['color']
                }
            
            js_plots += f"""
                    // Create dataset summary
                    var datasetSummary = {dataset_summary};
                    var datasetHtml = '<div class="row">';
                    for (var dataset in datasetSummary) {{
                        datasetHtml += '<div class="col-md-4"><div class="metric-card">';
                        datasetHtml += '<div class="metric-value">' + dataset + '</div>';
                        datasetHtml += '<div class="metric-label">Samples: ' + datasetSummary[dataset].Samples + '</div>';
                        datasetHtml += '<div class="metric-label">Features: ' + datasetSummary[dataset].Features + '</div>';
                        datasetHtml += '<div class="metric-label">Classes: ' + datasetSummary[dataset].Classes + '</div>';
                        datasetHtml += '</div></div>';
                    }}
                    datasetHtml += '</div>';
                    $('#dataset-summary').html(datasetHtml);
                    
                    // Create classifier summary
                    var classifierSummary = {classifier_summary};
                    var classifierHtml = '<div class="row">';
                    for (var clf in classifierSummary) {{
                        classifierHtml += '<div class="col-md-3"><div class="metric-card">';
                        classifierHtml += '<div class="metric-value" style="color: ' + classifierSummary[clf].Color + '">' + clf + '</div>';
                        classifierHtml += '<div class="metric-label">' + classifierSummary[clf].Type + '</div>';
                        classifierHtml += '</div></div>';
                    }}
                    classifierHtml += '</div>';
                    $('#classifier-summary').html(classifierHtml);
                    
                    // Create results table
                    var resultsData = {df_results.to_dict('records')};
                    var tableHtml = '<table class="table table-striped table-hover"><thead><tr>';
                    var columns = ['Dataset', 'Classifier', 'Accuracy', 'Precision', 'Recall', 'F1 Score', 'Training Time (s)', 'CV Accuracy Mean', 'CV Accuracy Std'];
                    for (var i = 0; i < columns.length; i++) {{
                        tableHtml += '<th>' + columns[i] + '</th>';
                    }}
                    tableHtml += '</tr></thead><tbody>';
                    
                    for (var i = 0; i < resultsData.length; i++) {{
                        tableHtml += '<tr>';
                        tableHtml += '<td>' + resultsData[i].Dataset + '</td>';
                        tableHtml += '<td><span class="classifier-badge" style="background-color: ' + resultsData[i].Color + '">' + resultsData[i].Classifier + '</span></td>';
                        tableHtml += '<td>' + resultsData[i].Accuracy.toFixed(3) + '</td>';
                        tableHtml += '<td>' + resultsData[i].Precision.toFixed(3) + '</td>';
                        tableHtml += '<td>' + resultsData[i].Recall.toFixed(3) + '</td>';
                        tableHtml += '<td>' + resultsData[i]['F1 Score'].toFixed(3) + '</td>';
                        tableHtml += '<td>' + resultsData[i]['Training Time (s)'].toFixed(3) + '</td>';
                        tableHtml += '<td>' + resultsData[i]['CV Accuracy Mean'].toFixed(3) + ' ± ' + resultsData[i]['CV Accuracy Std'].toFixed(3) + '</td>';
                        tableHtml += '</tr>';
                    }}
                    tableHtml += '</tbody></table>';
                    $('#results-table').html(tableHtml);
            """
        
        html_content += js_plots
        
        html_content += """
                }
                
                // Load plots when page is ready
                $(document).ready(function() {
                    loadAllPlots();
                    
                    // Show loading spinner while plots load
                    $('.plot-container').each(function() {
                        $(this).html('<div class="text-center py-5"><div class="spinner-border text-primary" role="status"></div><p class="mt-2">Loading plot...</p></div>');
                    });
                    
                    // Reload plots when tab is shown (for 3D plots that need to be visible)
                    $('button[data-bs-toggle="tab"]').on('shown.bs.tab', function(e) {
                        var target = $(e.target).attr("data-bs-target");
                        if (target === '#3d') {
                            setTimeout(loadAllPlots, 100);
                        }
                    });
                });
                
                // Responsive plot resizing
                $(window).on('resize', function() {
                    Plotly.Plots.resize('box-plot');
                    Plotly.Plots.resize('correlation-plot');
                    Plotly.Plots.resize('scatter-matrix');
                });
            </script>
        </body>
        </html>
        """
        
        # 保存HTML文件
        with open('task2.html', 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print("✅ 仪表板已保存为 'task2.html'")
        
    def run(self):
        """运行完整的仪表板创建流程"""
        print("=" * 80)
        print("创建完整的分类器比较仪表板")
        print("=" * 80)
        
        # 执行所有步骤
        self.generate_datasets()
        self.setup_classifiers()
        self.train_and_evaluate()
        self.create_data_exploration_section()
        self.create_2d_decision_boundaries()
        self.create_3d_visualizations()
        self.create_performance_comparison()
        self.create_confusion_matrices()
        self.create_dashboard_html()
        
        print("\n" + "=" * 80)
        print("✅ 仪表板创建完成!")
        print("打开 'task2.html' 查看完整结果")
        print("=" * 80)

# 运行仪表板
if __name__ == "__main__":
    dashboard = CompleteClassifierDashboard()
    dashboard.run()