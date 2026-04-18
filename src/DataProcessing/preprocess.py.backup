import os
import sys
import logging
import argparse
import numpy as np
import pandas as pd
import yaml
from sklearn.preprocessing import StandardScaler
import joblib
from pathlib import Path

# =======================
# 路径配置导入
# =======================
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

from configs.path_config import PROCESSED_DATA_DIR, ALIGNED_DATA_DIR, ROOT_DIR
from configs.path_config import get_path

# =======================
# 日志配置
# =======================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DatasetPreparator:
    def __init__(self, config_path):
        """
        初始化预处理类
        :param config_path: YAML 配置文件路径
        """
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"配置文件不存在: {config_path}")
            
        logger.info(f"加载配置文件: {config_path}")
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        # 提取数据处理相关配置
        if 'data_processing' in self.config:
            self.dp_config = self.config['data_processing']
        else:
            self.dp_config = self.config.get('data', self.config)
            
        # 关键参数提取
        self.feature_cols = self.dp_config['feature_cols']
        self.label_col = self.dp_config['label_col']
        self.time_col = self.dp_config['time_col']
        self.seq_len = self.dp_config['sequence_length']
        self.stride = self.dp_config.get('stride', 200)
        self.max_gap = self.dp_config.get('max_time_gap', 300)
        self.max_delta_soc = self.dp_config.get('max_delta_soc', 3.0)
        self.input_file = self.dp_config['input_file']
        
        self.exp_name = self.config.get('experiment_name', 'soc_model').replace(" ", "_")
        
        logger.info(f"配置加载完成:")
        logger.info(f"  - 序列长度 (Time Window): {self.seq_len}s")
        logger.info(f"  - 步长: {self.stride}")
        logger.info(f"  - 输入文件: {self.input_file}")
        logger.info(f"  - 实验前缀: {self.exp_name}")

    def load_data(self):
        """加载数据并进行初步清洗"""
        possible_paths = [
            ALIGNED_DATA_DIR / self.input_file,
            ROOT_DIR / self.input_file,
            Path(self.input_file) if os.path.isabs(self.input_file) else None
        ]
        
        file_path = None
        for p in possible_paths:
            if p and p.exists():
                file_path = p
                break

        if file_path is None:
            raise FileNotFoundError(f"未找到输入文件: {self.input_file}。搜索路径: {[str(p) for p in possible_paths if p]}")

        logger.info(f"正在加载数据: {file_path}")
        df = pd.read_csv(file_path)
        
        initial_rows = len(df)
        df = df.replace([np.inf, -np.inf], np.nan)
        nan_counts = df.isnull().sum()
        
        if nan_counts.any():
            logger.warning(f"发现缺失值:\n{nan_counts[nan_counts > 0]}")
            df = df.dropna()
            dropped = initial_rows - len(df)
            logger.info(f"已删除 {dropped} 行包含缺失值的数据。剩余行数: {len(df)}")
        
        if len(df) == 0:
            raise ValueError("清洗后数据为空！")

        df[self.time_col] = pd.to_datetime(df[self.time_col])
        sort_cols = ['trip_id', self.time_col] if 'trip_id' in df.columns else [self.time_col]
        df = df.sort_values(by=sort_cols).reset_index(drop=True)
            
        logger.info(f"数据加载与初步清洗完成。总行数: {len(df)}")
        return df

    def create_sequences(self, df):
        """构建时序样本"""
        logger.info(f"正在构建时序样本...")
        logger.info(f"  - 窗口长度 (Window): {self.seq_len} 步")
        logger.info(f"  - 采样步长 (Stride): {self.stride}")
        logger.info(f"  - 最大允许时间间隙 (Max Gap): {self.max_gap} 秒")
        logger.info(f"  - 最大允许 SOC 跳变 (Max Delta): {self.max_delta_soc}%")
        
        timestamps = df[self.time_col].values.astype(np.datetime64).astype(np.int64) // 10**9
        features = df[self.feature_cols].values
        labels = df[self.label_col].values
        
        n_samples = len(df)
        X_list = []
        y_list = []
        
        broken_windows = 0
        time_gap_filtered = 0
        outlier_filtered = 0
        
        max_delta_info = {'val': -np.inf, 'start_idx': -1, 'time_span': 0}
        min_delta_info = {'val': np.inf, 'start_idx': -1, 'time_span': 0}
        
        limit = n_samples - self.seq_len
        
        logger.info(f"开始扫描数据 (总行数: {n_samples})...")

        for i in range(0, limit, self.stride):
            end_idx = i + self.seq_len
            
            if end_idx >= n_samples:
                break
            
            time_span = timestamps[end_idx-1] - timestamps[i]
            
            if time_span > self.max_gap:
                time_gap_filtered += 1
                continue
            
            x_seq = features[i : end_idx] 
            
            try:
                start_soc = labels[i]
                end_soc = labels[end_idx-1]
                delta_soc = end_soc - start_soc
            except IndexError:
                broken_windows += 1
                continue

            if abs(delta_soc) > self.max_delta_soc:
                outlier_filtered += 1
                continue

            if np.isnan(x_seq).any() or np.isnan(delta_soc):
                broken_windows += 1
                continue
            
            if delta_soc > max_delta_info['val']:
                max_delta_info = {'val': delta_soc, 'start_idx': i, 'time_span': time_span}
            if delta_soc < min_delta_info['val']:
                min_delta_info = {'val': delta_soc, 'start_idx': i, 'time_span': time_span}
                
            X_list.append(x_seq)
            y_list.append(delta_soc)
            
        if len(X_list) == 0:
            raise ValueError("❌ 未生成任何有效样本。请检查 max_time_gap 或 max_delta_soc 设置是否过严。")
            
        y_arr = np.array(y_list)
        
        logger.info("="*50)
        logger.info("✅ 样本构建完成 - 统计报告")
        logger.info("="*50)
        logger.info(f"原始数据行数：{n_samples}")
        logger.info(f"生成有效样本数：{len(X_list)}")
        logger.info(f"  - 因时间断点 (> {self.max_gap}s) 丢弃：{time_gap_filtered}")
        logger.info(f"  - 因 SOC 跳变异常 (> {self.max_delta_soc}%) 丢弃：{outlier_filtered}")
        logger.info(f"  - 因数据缺失/错误 丢弃：{broken_windows}")
        logger.info("-"*50)
        logger.info("📈 处理后片段电量变化量 (Delta SOC) 分布分析:")
        logger.info(f"  - 最小值 (Min): {y_arr.min():.6f} %")
        logger.info(f"  - 最大值 (Max): {y_arr.max():.6f} %")
        logger.info(f"  - 平均值 (Mean): {y_arr.mean():.6f} %")
        logger.info(f"  - 标准差 (Std):  {y_arr.std():.6f} %")
        logger.info(f"  - 中位数 (Median): {np.median(y_arr):.6f} %")
        
        threshold_dynamic = y_arr.std() * 2
        dynamic_count = np.sum(np.abs(y_arr) > threshold_dynamic)
        dynamic_ratio = dynamic_count / len(y_arr) * 100
        
        logger.info(f"  - 剧烈变化样本 (|Δ| > {threshold_dynamic:.4f}%): {dynamic_count} 个 ({dynamic_ratio:.2f}%)")
        logger.info("="*50)
            
        return np.array(X_list), np.array(y_list)

    def _print_distribution_stats(self, name, data):
        """辅助函数：打印单组数据的分布统计"""
        data = np.array(data).flatten()
        count = len(data)
        mean_val = data.mean()
        std_val = data.std()
        min_val = data.min()
        max_val = data.max()
        
        # 关键指标：接近 0 的比例 (会导致 MAPE 爆炸)
        near_zero_thresh = 0.05 # 0.05% 阈值
        near_zero_count = np.sum(np.abs(data) < near_zero_thresh)
        near_zero_ratio = near_zero_count / count * 100
        
        # 关键指标：大变化比例
        large_thresh = 0.5 # 0.5% 阈值，可根据实际情况调整
        large_count = np.sum(np.abs(data) > large_thresh)
        large_ratio = large_count / count * 100
        
        logger.info(f"{name:10s} | Count: {count:6d} | Mean: {mean_val:7.4f} | Std: {std_val:7.4f} | "
                    f"Min: {min_val:7.4f} | Max: {max_val:7.4f} | "
                    f"Near-Zero(<0.05%): {near_zero_ratio:5.1f}% | Large(>0.5%): {large_ratio:5.1f}%")
        
        return {
            'mean': mean_val, 'std': std_val, 
            'near_zero_ratio': near_zero_ratio, 'large_ratio': large_ratio
        }

    def split_and_normalize(self, X, y):
        """按时间顺序切分并标准化 (新增分布检查 + 强制打乱)"""
        n = len(X)
        
        # =======================
        # 🔥 关键修复：随机打乱样本
        # =======================
        logger.info("⚠️ 检测到工况分布随时间变化显著，正在执行随机打乱 (Shuffle) 以混合工况...")
        indices = np.arange(n)
        np.random.seed(42)  # 固定种子保证可复现
        np.random.shuffle(indices)
        
        X = X[indices]
        y = y[indices]
        logger.info("✅ 数据已打乱。")
        # =======================
        
        train_cfg = self.config.get('model_training', {})
        val_split = train_cfg.get('validation_split', 0.2)
        test_split = train_cfg.get('test_split', 0.2)
        
        train_ratio = 1.0 - val_split - test_split
        val_ratio = val_split
        
        t_end = int(n * train_ratio)
        v_end = int(n * (train_ratio + val_ratio))
        
        if t_end == 0: t_end = 1
        if v_end <= t_end: v_end = t_end + 1
        if v_end >= n: v_end = n - 1
        
        # 切分 (此时已经是混合分布了)
        X_train, X_val, X_test = X[:t_end], X[t_end:v_end], X[v_end:]
        y_train, y_val, y_test = y[:t_end], y[t_end:v_end], y[v_end:]
        
        logger.info(f"数据切分: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")
        
        # =======================
        # 🚨 数据分布一致性检查
        # =======================
        logger.info("\n" + "="*60)
        logger.info("🚨 🚨 🚨 数据分布一致性检查 (原始标签 Delta SOC) 🚨 🚨 🚨")
        logger.info("="*60)
        
        # 确保变量被正确赋值
        stats_train = self._print_distribution_stats("Train", y_train)
        stats_val   = self._print_distribution_stats("Val",   y_val)
        stats_test  = self._print_distribution_stats("Test",  y_test)
        
        logger.info("-"*60)
        
        # 自动诊断警告 (增加安全性检查，防止变量未定义)
        warnings_found = False
        
        if stats_train and stats_test:
            # 1. 检查均值漂移
            mean_diff_ratio = abs(stats_test['mean'] - stats_train['mean']) / (abs(stats_train['mean']) + 1e-6)
            if mean_diff_ratio > 0.5: 
                logger.warning(f"⚠️ 严重警告：测试集均值 ({stats_test['mean']:.4f}) 与训练集 ({stats_train['mean']:.4f}) 差异巨大！")
                logger.warning(f"   相对差异：{mean_diff_ratio*100:.1f}%。")
                warnings_found = True
                
            # 2. 检查近零样本比例差异
            if 'near_zero_ratio' in stats_test and 'near_zero_ratio' in stats_train:
                zero_diff = abs(stats_test['near_zero_ratio'] - stats_train['near_zero_ratio'])
                if zero_diff > 20.0: 
                    logger.warning(f"⚠️ 严重警告：测试集近零样本比例 ({stats_test['near_zero_ratio']:.1f}%) 与训练集 ({stats_train['near_zero_ratio']:.1f}%) 差异巨大！")
                    logger.warning(f"   这将直接导致测试集 MAPE 指标爆炸。")
                    warnings_found = True
                
            # 3. 检查大变化样本比例
            if 'large_ratio' in stats_test and 'large_ratio' in stats_train:
                if stats_test['large_ratio'] < stats_train['large_ratio'] * 0.5:
                    logger.warning(f"⚠️ 警告：测试集剧烈变化样本占比 ({stats_test['large_ratio']:.1f}%) 远低于训练集 ({stats_train['large_ratio']:.1f}%)。")
                    warnings_found = True
        else:
            logger.error("❌ 统计信息获取失败，跳过分布警告检查。")

        if not warnings_found:
            logger.info("✅ 分布检查通过：Train/Val/Test 分布较为一致。")
        else:
            logger.warning("⚠️ 检测到分布差异，但由于已执行 Shuffle，这通常是随机波动，可继续训练。")
            
        logger.info("="*60 + "\n")
        
        # =======================
        # 标准化处理
        # =======================
        def safe_standardize(train_data, transform_data, name):
            scaler = StandardScaler()
            train_flat = train_data.reshape(-1, train_data.shape[-1])
            train_flat = np.nan_to_num(train_flat, nan=0.0, posinf=1e6, neginf=-1e6)
            
            scaler.fit(train_flat)
            
            if np.any(scaler.scale_ == 0):
                logger.warning(f"{name} 中存在标准差为 0 的特征。")
            
            t_flat = transform_data.reshape(-1, transform_data.shape[-1])
            t_flat = np.nan_to_num(t_flat, nan=0.0, posinf=1e6, neginf=-1e6)
            
            transformed = scaler.transform(t_flat).reshape(transform_data.shape)
            return transformed, scaler

        X_train_n, feat_scaler = safe_standardize(X_train, X_train, "Features (Train)")
        X_val_n, _ = safe_standardize(X_train, X_val, "Features (Val)")
        X_test_n, _ = safe_standardize(X_train, X_test, "Features (Test)")
        
        # 标签标准化
        label_scaler = StandardScaler()
        y_train_safe = np.nan_to_num(y_train.reshape(-1, 1), nan=0.0, posinf=1e6, neginf=-1e6)
        label_scaler.fit(y_train_safe)
        
        y_train_n = label_scaler.transform(y_train.reshape(-1, 1)).flatten()
        y_val_n = label_scaler.transform(y_val.reshape(-1, 1)).flatten()
        y_test_n = label_scaler.transform(y_test.reshape(-1, 1)).flatten()
        
        for name, arr in [("X_train", X_train_n), ("y_train", y_train_n)]:
            if np.isnan(arr).any() or np.isinf(arr).any():
                raise RuntimeError(f"致命错误：标准化后 {name} 仍包含 NaN/Inf!")

        logger.info("标准化完成。")
        
        return (X_train_n, X_val_n, X_test_n), \
               (y_train_n, y_val_n, y_test_n), \
               feat_scaler, label_scaler
        
        def safe_standardize(train_data, transform_data, name):
            scaler = StandardScaler()
            train_flat = train_data.reshape(-1, train_data.shape[-1])
            train_flat = np.nan_to_num(train_flat, nan=0.0, posinf=1e6, neginf=-1e6)
            
            scaler.fit(train_flat)
            
            if np.any(scaler.scale_ == 0):
                logger.warning(f"{name} 中存在标准差为 0 的特征。")
            
            t_flat = transform_data.reshape(-1, transform_data.shape[-1])
            t_flat = np.nan_to_num(t_flat, nan=0.0, posinf=1e6, neginf=-1e6)
            
            transformed = scaler.transform(t_flat).reshape(transform_data.shape)
            return transformed, scaler

        X_train_n, feat_scaler = safe_standardize(X_train, X_train, "Features (Train)")
        X_val_n, _ = safe_standardize(X_train, X_val, "Features (Val)")
        X_test_n, _ = safe_standardize(X_train, X_test, "Features (Test)")
        
        # 标签标准化
        label_scaler = StandardScaler()
        y_train_safe = np.nan_to_num(y_train.reshape(-1, 1), nan=0.0, posinf=1e6, neginf=-1e6)
        label_scaler.fit(y_train_safe)
        
        y_train_n = label_scaler.transform(y_train.reshape(-1, 1)).flatten()
        y_val_n = label_scaler.transform(y_val.reshape(-1, 1)).flatten()
        y_test_n = label_scaler.transform(y_test.reshape(-1, 1)).flatten()
        
        for name, arr in [("X_train", X_train_n), ("y_train", y_train_n)]:
            if np.isnan(arr).any() or np.isinf(arr).any():
                raise RuntimeError(f"致命错误：标准化后 {name} 仍包含 NaN/Inf!")

        logger.info("标准化完成。")
        
        return (X_train_n, X_val_n, X_test_n), \
               (y_train_n, y_val_n, y_test_n), \
               feat_scaler, label_scaler

    def save_data(self, X_tuple, y_tuple, scalers):
        """保存为 npz 和 pkl"""
        PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
        
        data_filename = f"{self.exp_name}_data.npz"
        output_path = PROCESSED_DATA_DIR / data_filename
        
        scaler_filename = f"{self.exp_name}_scalers.pkl"
        scaler_path = PROCESSED_DATA_DIR / scaler_filename
        
        X_train, X_val, X_test = X_tuple
        y_train, y_val, y_test = y_tuple
        feat_scaler, label_scaler = scalers
        
        np.savez(
            output_path,
            X_train=X_train, y_train=y_train,
            X_val=X_val, y_val=y_val,
            X_test=X_test, y_test=y_test
        )
        
        joblib.dump({'feature': feat_scaler, 'label': label_scaler}, scaler_path)
        
        logger.info(f"✅ 数据已保存至：{output_path}")
        logger.info(f"✅ Scaler 已保存至：{scaler_path}")
        
        print(f"\n=== 预处理成功 ===")
        print(f"配置序列长度：{self.seq_len}s")
        print(f"输出文件：{data_filename}")
        print(f"输入形状 (Train): {X_train.shape}")
        print(f"输出形状 (Train): {y_train.shape}")
        print(f"训练集 Label (标准化后) 均值：{y_train.mean():.4f}, 标准差：{y_train.std():.4f}")

def main():
    parser = argparse.ArgumentParser(description="SOC 数据预处理脚本 (支持多配置)")
    parser.add_argument('--config', type=str, default=None, 
                        help='YAML 配置文件路径')
    args = parser.parse_args()
    
    config_path = args.config
    if config_path is None:
        default_paths = [
            get_path("config"),
            os.path.join(project_root, "configs", "config.yaml"),
            os.path.join(project_root, "config.yaml")
        ]
        for p in default_paths:
            if p and os.path.exists(p):
                config_path = p
                logger.warning(f"未指定 --config，使用默认路径：{config_path}")
                break
        
        if config_path is None or not os.path.exists(config_path):
            logger.error("❌ 错误：未找到配置文件。请使用 --config 参数指定 YAML 文件路径。")
            sys.exit(1)

    try:
        preparator = DatasetPreparator(config_path)
        df = preparator.load_data()
        X, y = preparator.create_sequences(df)
        
        # 这里将执行分布检查
        X_tuple, y_tuple, feat_scaler, label_scaler = preparator.split_and_normalize(X, y)
        
        scalers = (feat_scaler, label_scaler) 
        preparator.save_data(X_tuple, y_tuple, scalers)
        
        logger.info("🎉 全部处理完成！")
        
    except Exception as e:
        logger.error(f"❌ 处理过程中发生错误：{e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()