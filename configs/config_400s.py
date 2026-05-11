"""
时间片段划分配置文件 - 400秒核心版本
"""

import os
from pathlib import Path

# 基础配置
class TimeSegmentConfig:
    """时间片段划分配置类"""

    # 时间片段参数
    CORE_STEPS = 40          # 核心时间步数
    CONTEXT_STEPS = 0        # 上下文时间步数（前后各1）
    TOTAL_STEPS = 40

    # 时间参数（秒）
    TIME_RESOLUTION = 10     # 时间分辨率：10秒/步
    CORE_SECONDS = CORE_STEPS * TIME_RESOLUTION  # 核心时长：400秒
    BREAK_THRESHOLD = CORE_SECONDS + 100  # 连续段断点阈值：500秒
    MAX_SEGMENT_DURATION = CORE_SECONDS + 100  # 片段最大时长：500秒

    # 步长参数
    STEP_SIZE = CORE_STEPS // 2  # 步长 = 核心的一半 = 20

    # 输入输出路径
    INPUT_DATA_PATH = Path("data/processed/feature_data.csv")
    OUTPUT_DIR = Path("data/split")
    OUTPUT_FILE = OUTPUT_DIR / "origin_400s.npz"

    # 数据列配置
    TIME_COLUMN = "DATE"  # 时间列名
    REQUIRED_COLUMNS = [
        "DATE", "speed", "mileage", "total_volt", "total_current",
        "standard_soc", "temperature_c", "relative_humidity", "visibility_km", "wind_speed_ms",
        "time_diff", "is_new_trip", "trip_id", "current_discharge", "delta_q_ah",
        "refined_soc", "speed_category", "is_run", "speed_diff", "mileage_diff",
        "power", "window_start", "window_end", "window_size",
        "speed_window20_mean", "speed_diff_window20_mean", "temperature_c_window20_mean",
        "relative_humidity_window20_mean", "visibility_km_window20_mean", "wind_speed_ms_window20_mean",
        "speed_window20_std", "Low", "Mid", "High", "cruising_ratio"
    ]# 35列

    # 处理参数
    CHUNK_SIZE = 10000  # 分批处理大小
    VERBOSE = True      # 是否显示详细日志

    def __init__(self):
        """初始化配置，确保输出目录存在"""
        self.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    def validate(self):
        """验证配置参数合理性"""
        assert self.CORE_STEPS > 0, "核心步数必须大于0"
        assert self.CONTEXT_STEPS >= 0, "上下文步数不能为负"
        assert self.TOTAL_STEPS == self.CORE_STEPS + 2 * self.CONTEXT_STEPS, "总步数计算错误"
        assert self.TIME_RESOLUTION > 0, "时间分辨率必须大于0"
        assert self.BREAK_THRESHOLD > 0, "断点阈值必须大于0"
        assert self.MAX_SEGMENT_DURATION > 0, "最大片段时长必须大于0"
        assert self.STEP_SIZE > 0, "步长必须大于0"
        assert self.CHUNK_SIZE > 0, "分批大小必须大于0"

        print(f"配置验证通过:")
        print(f"  核心步数: {self.CORE_STEPS}步 ({self.CORE_SECONDS}秒)")
        print(f"  上下文步数: 前后各{self.CONTEXT_STEPS}步")
        print(f"  总步数: {self.TOTAL_STEPS}步")
        print(f"  步长: {self.STEP_SIZE}步")
        print(f"  断点阈值: {self.BREAK_THRESHOLD}秒")
        print(f"  片段最大时长: {self.MAX_SEGMENT_DURATION}秒")
        print(f"  输入文件: {self.INPUT_DATA_PATH}")
        print(f"  输出文件: {self.OUTPUT_FILE}")

        return True

# 创建配置实例
config = TimeSegmentConfig()

if __name__ == "__main__":
    # 测试配置
    config.validate()