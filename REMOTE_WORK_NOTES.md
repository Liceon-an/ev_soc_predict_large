# 远程工作注意事项
## ev_soc_predict 项目远程开发指南

### 1. 项目位置
- **远程服务器**: AutoDL服务器 (无GPU状态)
- **SSH地址**: `root@connect.westc.seetacloud.com:31720`
- **项目路径**: `/root/code/ev_soc_predict/`
- **本地文档**: `/Users/liceon/prompt/` (仅文档，无代码)

### 2. SSH连接命令
```bash
# 基本连接
ssh -o ConnectTimeout=10 root@connect.westc.seetacloud.com -p 31720

# 带命令执行（单条命令）
ssh -o ConnectTimeout=10 root@connect.westc.seetacloud.com -p 31720 "cd /root/code/ev_soc_predict && ls -la"

# 文件上传
scp -P 31720 local_file.py root@connect.westc.seetacloud.com:/root/code/ev_soc_predict/path/

# 文件下载
scp -P 31720 root@connect.westc.seetacloud.com:/root/code/ev_soc_predict/path/file.py ./
```

### 3. 环境配置
**重要**: 服务器没有系统Python，必须使用conda环境：
```bash
# 激活conda环境
source /root/miniconda3/bin/activate

# 验证Python路径
which python  # 应该显示: /root/miniconda3/bin/python

# 项目依赖
cd /root/code/ev_soc_predict
pip install -r requirements.txt
```

### 4. 常用命令参考

#### 4.1 项目导航
```bash
# 进入项目目录
cd /root/code/ev_soc_predict

# 查看项目结构
ls -la
find src -name "*.py" | sort

# 查看配置文件
ls -la configs/
cat configs/config_enhanced.yaml
```

#### 4.2 新架构测试命令
```bash
# 激活环境并进入项目
source /root/miniconda3/bin/activate
cd /root/code/ev_soc_predict

# 测试各个模块
python -c "from src.DataProcessing.continuity_detector import ContinuityDetector; print('OK')"
python -c "from src.DataProcessing.time_segment_splitter import TimeSegmentSplitter; print('OK')"
python -c "from src.FeatureEngineering.enhanced_feature_engineer import process_time_segment; print('OK')"
python -c "from src.DataProcessing.enhanced_preprocessor import EnhancedDataPreparator; print('OK')"

# 运行增强预处理器测试
python src/DataProcessing/enhanced_preprocessor.py --config configs/config_enhanced.yaml --output data/processed/test
```

#### 4.3 数据检查
```bash
# 检查输入数据
ls -la data/
head -5 data/aligned_data_refined_soc.csv

# 检查已有处理结果
ls -la data/processed/
```

### 5. 遇到的问题与解决方案

#### 5.1 Python命令不存在
**问题**: `bash: python: command not found`
**解决**: 使用conda环境中的Python
```bash
# 错误
python script.py

# 正确
/root/miniconda3/bin/python script.py
# 或
source /root/miniconda3/bin/activate && python script.py
```

#### 5.2 导入错误
**问题**: `ImportError: cannot import name 'setup_logging'`
**原因**: `src/utils.py`中没有这些函数
**解决**: 已在`enhanced_preprocessor.py`中直接实现

#### 5.3 SSH heredoc编码问题
**问题**: 通过SSH使用heredoc创建文件时出现乱码
**解决**: 使用`scp`上传或简化命令
```bash
# 简化版本
ssh -p 31720 root@connect.westc.seetacloud.com "echo '内容' > /path/file.py"
```

### 6. 新架构文件清单
以下文件是2026-04-18新创建/完善的：

#### 6.1 新模块文件
- `src/FeatureEngineering/enhanced_feature_engineer.py` - 增强特征工程
- `src/DataProcessing/continuity_detector.py` - 连续段检测器
- `src/DataProcessing/time_segment_splitter.py` - 时间片段划分器
- `src/DataProcessing/enhanced_preprocessor.py` - 增强预处理器

#### 6.2 配置文件
- `configs/config_enhanced.yaml` - 新架构专用配置

#### 6.3 备份文件
- `src/DataProcessing/enhanced_preprocessor.py.backup` - 原始备份

### 7. 下一步测试计划

#### 7.1 基本功能测试（已完成✅）
- [x] 模块导入测试
- [x] 类初始化测试
- [x] 配置加载测试

#### 7.2 完整流程测试（待进行）
1. **数据加载测试**: 使用实际CSV文件
2. **处理流程测试**: 运行完整的新架构流程
3. **输出验证**: 检查生成的.npz文件和元数据
4. **性能测试**: 对比新旧架构处理速度

#### 7.3 集成测试（待进行）
1. **模型兼容性**: 测试LSTM模型能否处理新特征
2. **训练流程**: 完整训练流程测试
3. **结果验证**: 预测结果准确性验证

### 8. 紧急联系信息
- **项目文档**: `/Users/liceon/prompt/ticket.md` (本地)
- **代码位置**: `/root/code/ev_soc_predict/` (远程)
- **最后更新**: 2026-04-18
- **当前状态**: 新架构代码完成，等待完整测试

### 9. 快速开始指南
```bash
# 1. 连接到服务器
ssh -p 31720 root@connect.westc.seetacloud.com

# 2. 设置环境
source /root/miniconda3/bin/activate
cd /root/code/ev_soc_predict

# 3. 运行新架构测试
python src/DataProcessing/enhanced_preprocessor.py \
  --config configs/config_enhanced.yaml \
  --output data/processed/first_test \
  --log-level INFO
```

---
**备注**: 本文件记录了远程工作的关键信息和常见问题解决方案。建议在开始工作前阅读ticket.md获取完整的项目上下文。