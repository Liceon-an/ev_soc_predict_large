'''
时间片段划分配置文件 - 1500秒核心版本
'''

from pathlib import Path


class TimeSegmentConfig1500:
    CORE_STEPS = 150
    CONTEXT_STEPS = 0
    TOTAL_STEPS = 150

    TIME_RESOLUTION = 10
    CORE_SECONDS = CORE_STEPS * TIME_RESOLUTION
    BREAK_THRESHOLD = CORE_SECONDS + 100
    MAX_SEGMENT_DURATION = CORE_SECONDS + 100
    STEP_SIZE = CORE_STEPS // 2

    INPUT_DATA_PATH = Path('data/processed/feature_data.csv')
    OUTPUT_DIR = Path('data/split')
    OUTPUT_FILE = OUTPUT_DIR / 'origin_1500s.npz'

    TIME_COLUMN = 'DATE'
    REQUIRED_COLUMNS = [
        'DATE', 'speed', 'mileage', 'total_volt', 'total_current',
        'standard_soc', 'temperature_c', 'relative_humidity', 'visibility_km', 'wind_speed_ms',
        'time_diff', 'is_new_trip', 'trip_id', 'current_discharge', 'delta_q_ah',
        'refined_soc', 'speed_category', 'is_run', 'speed_diff', 'mileage_diff',
        'power', 'window_start', 'window_end', 'window_size',
        'speed_window20_mean', 'speed_diff_window20_mean', 'temperature_c_window20_mean',
        'relative_humidity_window20_mean', 'visibility_km_window20_mean', 'wind_speed_ms_window20_mean',
        'speed_window20_std', 'Low', 'Mid', 'High', 'cruising_ratio'
    ]

    CHUNK_SIZE = 10000
    VERBOSE = True

    def __init__(self):
        self.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    def validate(self):
        assert self.CORE_STEPS > 0
        assert self.CONTEXT_STEPS >= 0
        assert self.TOTAL_STEPS == self.CORE_STEPS + 2 * self.CONTEXT_STEPS
        assert self.TIME_RESOLUTION > 0
        assert self.BREAK_THRESHOLD > 0
        assert self.MAX_SEGMENT_DURATION > 0
        assert self.STEP_SIZE > 0
        assert self.CHUNK_SIZE > 0
        print(f'配置验证通过: 核心步数={self.CORE_STEPS}步 ({self.CORE_SECONDS}秒)')
        return True


config = TimeSegmentConfig1500()

if __name__ == '__main__':
    config.validate()
