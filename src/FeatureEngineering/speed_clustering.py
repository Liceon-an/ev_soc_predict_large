#!/usr/bin/env python3
"""
速度聚类分析工具
分析和可视化速度聚类结果
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import warnings
warnings.filterwarnings('ignore')

class SpeedClusteringAnalyzer:
    """
    速度聚类分析器
    分析速度数据的聚类特征，帮助确定最佳聚类数量
    """
    
    def __init__(self, speeds: np.ndarray):
        """
        初始化分析器
        
        Args:
            speeds: 速度数组
        """
        self.speeds = speeds.copy()
        self.valid_speeds = speeds[~np.isnan(speeds)].reshape(-1, 1)
        
        if len(self.valid_speeds) == 0:
            raise ValueError("没有有效的速度数据")
        
        self.cluster_results = {}
        self.best_n_clusters = None
    
    def analyze_cluster_range(self, min_clusters=2, max_clusters=10):
        """
        分析不同聚类数量的效果
        
        Args:
            min_clusters: 最小聚类数
            max_clusters: 最大聚类数
        """
        cluster_range = range(min_clusters, max_clusters + 1)
        
        results = []
        for n_clusters in cluster_range:
            if len(self.valid_speeds) < n_clusters:
                print(f"警告: 数据点数量({len(self.valid_speeds)})少于聚类数({n_clusters})，跳过")
                continue
            
            try:
                # 执行K-means聚类
                kmeans = KMeans(
                    n_clusters=n_clusters,
                    random_state=42,
                    n_init=10,
                    max_iter=300
                )
                labels = kmeans.fit_predict(self.valid_speeds)
                centers = kmeans.cluster_centers_.flatten()
                
                # 计算评估指标
                if len(np.unique(labels)) > 1:  # 需要至少2个聚类才能计算某些指标
                    silhouette = silhouette_score(self.valid_speeds, labels)
                    calinski = calinski_harabasz_score(self.valid_speeds, labels)
                    davies = davies_bouldin_score(self.valid_speeds, labels)
                else:
                    silhouette = calinski = davies = np.nan
                
                # 计算聚类统计
                cluster_stats = []
                for i in range(n_clusters):
                    cluster_speeds = self.valid_speeds[labels == i]
                    if len(cluster_speeds) > 0:
                        stats = {
                            'cluster': i,
                            'center': centers[i],
                            'count': len(cluster_speeds),
                            'min': float(cluster_speeds.min()),
                            'max': float(cluster_speeds.max()),
                            'mean': float(cluster_speeds.mean()),
                            'std': float(cluster_speeds.std())
                        }
                        cluster_stats.append(stats)
                
                # 按中心排序
                cluster_stats.sort(key=lambda x: x['center'])
                
                # 计算阈值（聚类中心之间的中点）
                thresholds = []
                for i in range(len(cluster_stats) - 1):
                    threshold = (cluster_stats[i]['center'] + cluster_stats[i + 1]['center']) / 2
                    thresholds.append(threshold)
                
                result = {
                    'n_clusters': n_clusters,
                    'labels': labels,
                    'centers': centers,
                    'silhouette': silhouette,
                    'calinski_harabasz': calinski,
                    'davies_bouldin': davies,
                    'cluster_stats': cluster_stats,
                    'thresholds': thresholds,
                    'model': kmeans
                }
                
                results.append(result)
                self.cluster_results[n_clusters] = result
                
                print(f"聚类数 {n_clusters}:")
                print(f"  轮廓系数: {silhouette:.4f}")
                print(f"  Calinski-Harabasz: {calinski:.2f}")
                print(f"  Davies-Bouldin: {davies:.4f}")
                print(f"  聚类中心: {np.sort(centers)}")
                print(f"  阈值: {thresholds}")
                print()
                
            except Exception as e:
                print(f"聚类数 {n_clusters} 失败: {e}")
        
        return results
    
    def find_optimal_clusters(self, method='silhouette'):
        """
        寻找最优聚类数量
        
        Args:
            method: 选择方法 ('silhouette', 'calinski', 'davies')
            
        Returns:
            最优聚类数量
        """
        if not self.cluster_results:
            self.analyze_cluster_range(2, 8)
        
        valid_results = {k: v for k, v in self.cluster_results.items() 
                        if not np.isnan(v.get('silhouette', np.nan))}
        
        if not valid_results:
            print("没有有效的聚类结果")
            return None
        
        if method == 'silhouette':
            # 轮廓系数越大越好
            best_n = max(valid_results.keys(), 
                        key=lambda k: valid_results[k]['silhouette'])
            best_score = valid_results[best_n]['silhouette']
            print(f"最优聚类数（轮廓系数）: {best_n} (得分: {best_score:.4f})")
            
        elif method == 'calinski':
            # Calinski-Harabasz指数越大越好
            best_n = max(valid_results.keys(),
                        key=lambda k: valid_results[k]['calinski_harabasz'])
            best_score = valid_results[best_n]['calinski_harabasz']
            print(f"最优聚类数（Calinski-Harabasz）: {best_n} (得分: {best_score:.2f})")
            
        elif method == 'davies':
            # Davies-Bouldin指数越小越好
            best_n = min(valid_results.keys(),
                        key=lambda k: valid_results[k]['davies_bouldin'])
            best_score = valid_results[best_n]['davies_bouldin']
            print(f"最优聚类数（Davies-Bouldin）: {best_n} (得分: {best_score:.4f})")
        
        self.best_n_clusters = best_n
        return best_n
    
    def visualize_clusters(self, n_clusters=None, save_path=None):
        """
        可视化聚类结果
        
        Args:
            n_clusters: 聚类数量，如果为None则使用最优聚类数
            save_path: 保存路径
        """
        if n_clusters is None:
            if self.best_n_clusters is None:
                self.find_optimal_clusters()
            n_clusters = self.best_n_clusters
        
        if n_clusters not in self.cluster_results:
            print(f"没有 {n_clusters} 个聚类的结果")
            return
        
        result = self.cluster_results[n_clusters]
        labels = result['labels']
        centers = result['centers']
        
        # 创建图形
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 1. 速度分布直方图
        ax = axes[0, 0]
        ax.hist(self.valid_speeds, bins=50, alpha=0.7, edgecolor='black')
        ax.set_xlabel('速度 (km/h)')
        ax.set_ylabel('频数')
        ax.set_title('速度分布直方图')
        ax.grid(True, alpha=0.3)
        
        # 添加聚类中心线
        colors = plt.cm.tab10(np.arange(n_clusters) / n_clusters)
        for i, center in enumerate(np.sort(centers)):
            ax.axvline(center, color=colors[i], linestyle='--', 
                      linewidth=2, label=f'聚类 {i} 中心: {center:.1f}')
        ax.legend()
        
        # 2. 聚类散点图
        ax = axes[0, 1]
        # 为了可视化，添加一些随机噪声在y轴上
        y_noise = np.random.randn(len(self.valid_speeds)) * 0.1
        scatter = ax.scatter(self.valid_speeds, y_noise, c=labels, 
                           cmap='tab10', alpha=0.6, s=20)
        ax.set_xlabel('速度 (km/h)')
        ax.set_yticks([])
        ax.set_title(f'速度聚类 (k={n_clusters})')
        ax.grid(True, alpha=0.3)
        
        # 添加聚类中心
        for i, center in enumerate(np.sort(centers)):
            ax.scatter([center], [0], color=colors[i], s=200, 
                      marker='X', edgecolor='black', linewidth=2)
        
        # 3. 聚类评估指标
        ax = axes[1, 0]
        cluster_nums = list(self.cluster_results.keys())
        silhouette_scores = [self.cluster_results[n]['silhouette'] for n in cluster_nums]
        calinski_scores = [self.cluster_results[n]['calinski_harabasz'] for n in cluster_nums]
        
        ax.plot(cluster_nums, silhouette_scores, 'o-', linewidth=2, label='轮廓系数')
        ax.set_xlabel('聚类数量')
        ax.set_ylabel('轮廓系数', color='blue')
        ax.tick_params(axis='y', labelcolor='blue')
        ax.grid(True, alpha=0.3)
        
        ax2 = ax.twinx()
        ax2.plot(cluster_nums, calinski_scores, 's-', linewidth=2, color='red', label='Calinski-Harabasz')
        ax2.set_ylabel('Calinski-Harabasz', color='red')
        ax2.tick_params(axis='y', labelcolor='red')
        
        # 标记最优聚类数
        if self.best_n_clusters:
            best_idx = cluster_nums.index(self.best_n_clusters)
            ax.axvline(self.best_n_clusters, color='green', linestyle=':', 
                      linewidth=2, label=f'最优: k={self.best_n_clusters}')
        
        ax.set_title('聚类评估指标')
        ax.legend(loc='upper left')
        ax2.legend(loc='upper right')
        
        # 4. 聚类统计信息
        ax = axes[1, 1]
        cluster_stats = result['cluster_stats']
        
        # 创建表格数据
        table_data = []
        for stats in cluster_stats:
            table_data.append([
                stats['cluster'],
                stats['count'],
                f"{stats['center']:.1f}",
                f"{stats['min']:.1f}",
                f"{stats['max']:.1f}",
                f"{stats['mean']:.1f}",
                f"{stats['std']:.1f}"
            ])
        
        # 创建表格
        column_labels = ['聚类', '样本数', '中心', '最小值', '最大值', '均值', '标准差']
        table = ax.table(cellText=table_data, colLabels=column_labels,
                        loc='center', cellLoc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 1.5)
        
        ax.axis('off')
        ax.set_title('聚类统计信息')
        
        plt.suptitle(f'速度聚类分析 (最优聚类数: {n_clusters})', fontsize=16)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"可视化结果已保存: {save_path}")
        
        plt.show()
        
        return fig
    
    def get_clustering_config(self, n_clusters=None):
        """
        获取聚类配置
        
        Args:
            n_clusters: 聚类数量
            
        Returns:
            聚类配置字典
        """
        if n_clusters is None:
            if self.best_n_clusters is None:
                self.find_optimal_clusters()
            n_clusters = self.best_n_clusters
        
        if n_clusters not in self.cluster_results:
            print(f"没有 {n_clusters} 个聚类的结果")
            return None
        
        result = self.cluster_results[n_clusters]
        
        config = {
            'n_clusters': n_clusters,
            'thresholds': result['thresholds'],
            'centers': result['centers'].tolist(),
            'cluster_stats': result['cluster_stats'],
            'evaluation': {
                'silhouette_score': float(result['silhouette']),
                'calinski_harabasz_score': float(result['calinski_harabasz']),
                'davies_bouldin_score': float(result['davies_bouldin'])
            }
        }
        
        return config

def analyze_real_data(data_path, sample_size=10000):
    """
    分析真实数据的速度聚类
    
    Args:
        data_path: 数据文件路径
        sample_size: 采样大小（如果数据太大）
    """
    print(f"加载数据: {data_path}")
    
    # 加载数据
    df = pd.read_csv(data_path)
    
    if 'speed' not in df.columns:
        print("数据中没有'speed'列")
        return
    
    # 采样（如果数据太大）
    if len(df) > sample_size:
        df = df.sample(sample_size, random_state=42)
        print(f"采样 {sample_size} 个数据点")
    
    speeds = df['speed'].values
    
    print(f"数据点数: {len(speeds)}")
    print(f"有效速度点数: {np.sum(~np.isnan(speeds))}")
    print(f"速度范围: [{np.nanmin(speeds):.1f}, {np.nanmax(speeds):.1f}]")
    print(f"速度均值: {np.nanmean(speeds):.1f}")
    print(f"速度标准差: {np.nanstd(speeds):.1f}")
    
    # 创建分析器
    analyzer = SpeedClusteringAnalyzer(speeds)
    
    # 分析聚类范围
    print("\n分析聚类效果:")
    analyzer.analyze_cluster_range(2, 8)
    
    # 寻找最优聚类数
    print("\n寻找最优聚类数:")
    best_n = analyzer.find_optimal_clusters('silhouette')
    
    # 可视化
    print("\n生成可视化...")
    fig = analyzer.visualize_clusters(best_n)
    
    # 获取配置
    config = analyzer.get_clustering_config(best_n)
    
    return analyzer, config

if __name__ == "__main__":
    import sys
    import os
    
    # 添加项目路径
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    sys.path.insert(0, project_root)
    
    # 测试数据路径
    data_path = os.path.join(project_root, 'data', 'aligned', 'aligned_data_refined_soc.csv')
    
    if os.path.exists(data_path):
        print("分析真实数据...")
        analyzer, config = analyze_real_data(data_path, sample_size=5000)
        
        if config:
            print("\n聚类配置:")
            print(f"最优聚类数: {config['n_clusters']}")
            print(f"阈值: {config['thresholds']}")
            print(f"聚类中心: {config['centers']}")
    else:
        print(f"数据文件不存在: {data_path}")
        print("使用模拟数据测试...")
        
        # 创建模拟数据
        np.random.seed(42)
        n_samples = 1000
        # 模拟三种速度模式：低速（0-20），中速（20-60），高速（60-100）
        low_speed = np.random.normal(10, 5, n_samples//3)
        medium_speed = np.random.normal(40, 10, n_samples//3)
        high_speed = np.random.normal(80, 15, n_samples//3)
        
        speeds = np.concatenate([low_speed, medium_speed, high_speed])
        speeds = np.clip(speeds, 0, 120)  # 限制在合理范围
        
        analyzer = SpeedClusteringAnalyzer(speeds)
        analyzer.analyze_cluster_range(2, 6)
        best_n = analyzer.find_optimal_clusters('silhouette')
        analyzer.visualize_clusters(best_n)
