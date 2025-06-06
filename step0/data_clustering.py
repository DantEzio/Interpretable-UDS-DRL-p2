import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import dendrogram, linkage

class DataClustering:
    def __init__(self, data):
        """
        初始化聚类分析类
        
        参数:
            data (DataFrame): 包含特征的DataFrame
        """
        self.data = data
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=2)
        self.clusters = None
        self.features = None
        self.silhouette_avg = None

        # 设置中文字体
        plt.rcParams["font.family"] = ["SimHei"]
        plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题
    
    def preprocess_data(self, feature_cols=None, scale=True, apply_pca=True):
        """
        数据预处理
        
        参数:
            feature_cols (list): 特征列名列表，默认为None(使用所有列)
            scale (bool): 是否标准化数据，默认为True
            apply_pca (bool): 是否应用PCA降维，默认为True
        """
        # 选择特征列
        if feature_cols:
            self.features = self.data[feature_cols].values
        else:
            self.features = self.data.values
        
        # 数据标准化
        if scale:
            self.features = self.scaler.fit_transform(self.features)
        
        # PCA降维
        if apply_pca and self.features.shape[1] > 2:
            self.features = self.pca.fit_transform(self.features)
            print(f"PCA解释方差比例: {self.pca.explained_variance_ratio_.sum():.2%}")
    
    def kmeans_clustering(self, n_clusters=3, random_state=42):
        """
        K-Means聚类
        
        参数:
            n_clusters (int): 聚类数量，默认为3
            random_state (int): 随机种子，默认为42
        """
        if self.features is None:
            self.preprocess_data()
        
        kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
        self.clusters = kmeans.fit_predict(self.features)
        self.silhouette_avg = silhouette_score(self.features, self.clusters)
        print(f"K-Means聚类完成，轮廓系数: {self.silhouette_avg:.4f}")
        
        return self.clusters
    
    def dbscan_clustering(self, eps=0.5, min_samples=5, leaf_size=10):
        """
        DBSCAN聚类
        
        参数:
            eps (float): 邻域半径，默认为0.5
            min_samples (int): 形成核心点所需的最小样本数，默认为5
        """
        if self.features is None:
            self.preprocess_data()
        
        dbscan = DBSCAN(eps=eps, min_samples=min_samples, leaf_size=leaf_size)
        self.clusters = dbscan.fit_predict(self.features)
        
        # 计算轮廓系数(排除噪声点)
        core_samples_mask = np.zeros_like(dbscan.labels_, dtype=bool)
        core_samples_mask[dbscan.core_sample_indices_] = True
        valid_indices = np.where(self.clusters != -1)[0]
        
        if len(valid_indices) > 10:  # 确保有足够的样本计算轮廓系数
            self.silhouette_avg = silhouette_score(
                self.features[valid_indices], 
                self.clusters[valid_indices]
            )
            print(f"DBSCAN聚类完成，轮廓系数: {self.silhouette_avg:.4f}")
        else:
            print("DBSCAN聚类完成，但无法计算轮廓系数(有效样本不足)")
        
        return self.clusters
    
    def hierarchical_clustering(self, method='ward', metric='euclidean'):
        """
        层次聚类
        
        参数:
            method (str): 链接方法，默认为'ward'
            metric (str): 距离度量，默认为'euclidean'
        """
        if self.features is None:
            self.preprocess_data()
        
        # 计算层次聚类的链接矩阵
        self.linkage_matrix = linkage(self.features, method=method, metric=metric)
        print(f"层次聚类链接矩阵计算完成，形状: {self.linkage_matrix.shape}")
        
        return self.linkage_matrix
    
    def plot_dendrogram(self, max_d=None, figsize=(10, 6)):
        """
        绘制层次聚类树状图
        
        参数:
            max_d (float): 截断距离，默认为None
            figsize (tuple): 图表大小，默认为(10, 6)
        """
        if not hasattr(self, 'linkage_matrix'):
            self.hierarchical_clustering()
        
        plt.figure(figsize=figsize)
        plt.title('层次聚类树状图')
        plt.xlabel('样本索引')
        plt.ylabel('距离')
        
        dendrogram(
            self.linkage_matrix,
            truncate_mode='lastp',  # 限制显示的叶节点数量
            p=30,                   # 显示最后p个合并
            show_contracted=True,   # 显示收缩信息
            color_threshold=max_d   # 截断距离
        )
        
        if max_d:
            plt.axhline(y=max_d, c='k', linestyle='--')
        
        plt.tight_layout()
        return plt
    
    def plot_clusters(self, figsize=(10, 8), title=None):
        """
        可视化聚类结果
        
        参数:
            figsize (tuple): 图表大小，默认为(10, 8)
            title (str): 图表标题，默认为None
        """
        if self.clusters is None:
            raise ValueError("请先执行聚类算法")
        
        plt.figure(figsize=figsize)
        
        # 获取聚类数量(排除噪声点)
        unique_labels = set(self.clusters)
        n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
        
        # 设置颜色映射
        colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_labels)))
        
        for k, col in zip(unique_labels, colors):
            if k == -1:
                # 噪声点用黑色表示
                col = 'k'
            
            class_mask = (self.clusters == k)
            plt.scatter(
                self.features[class_mask, 0], 
                self.features[class_mask, 1],
                s=50, c=[col], label=f'聚类 {k}',
                alpha=0.7, edgecolors='w'
            )
        
        plt.title(title or f'聚类结果 (n_clusters={n_clusters})')
        plt.xlabel('特征1')
        plt.ylabel('特征2')
        plt.legend(loc='best')
        plt.grid(True, linestyle='--', alpha=0.7)
        
        return plt

# 使用示例
if __name__ == "__main__":
    # 生成示例数据
    np.random.seed(42)
    data = pd.DataFrame({
        'x1': np.random.randn(300),
        'x2': np.random.randn(300)
    })
    
    # 添加聚类结构
    data.loc[:99, 'x1'] += 5
    data.loc[:99, 'x2'] += 5
    data.loc[100:199, 'x1'] -= 5
    data.loc[100:199, 'x2'] += 5
    
    # 创建聚类分析实例
    clustering = DataClustering(data)
    
    # 执行K-Means聚类
    clustering.preprocess_data(scale=True, apply_pca=False)
    kmeans_labels = clustering.kmeans_clustering(n_clusters=3)
    
    # 可视化K-Means结果
    plt_kmeans = clustering.plot_clusters(title='K-Means聚类结果')
    plt_kmeans.show()
    
    # 执行DBSCAN聚类
    dbscan_labels = clustering.dbscan_clustering(eps=0.5, min_samples=5)
    
    # 可视化DBSCAN结果
    plt_dbscan = clustering.plot_clusters(title='DBSCAN聚类结果')
    plt_dbscan.show()
    
    # 执行层次聚类并绘制树状图
    linkage_matrix = clustering.hierarchical_clustering()
    plt_dendro = clustering.plot_dendrogram(max_d=10)
    plt_dendro.show()    