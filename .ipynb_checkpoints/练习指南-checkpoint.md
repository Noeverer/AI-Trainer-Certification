# 📚 人工智能训练师三级 - 练习指南

> 🎯 高效练习方法 + 每类题目注意事项

---

## 🚀 高效练习方法

### 方法一：默写法（最有效）

1. **看代码** → 理解每行的作用
2. **合上资料** → 默写代码
3. **对照检查** → 找出错误
4. **重复** → 直到能完整默写

### 方法二：填空法（适合碎片时间）

把代码模板打印出来，遮住关键部分，口头回忆：
- 遮住 `import` 语句
- 遮住函数名和参数
- 遮住变量名

### 方法三：实战法（考前半天）

打开Jupyter，不看答案，尝试完成一道完整题目：
1. 打开「编程题_未填写答案」文件夹的题目
2. 计时完成
3. 对照「编程题_已填写答案」检查

---

## 📋 题目类型详解与练习要点

---

## 1️⃣ 1.1.X 数据分析统计题

### 题目特点
- 给你一个CSV数据文件
- 要求进行统计分析（计数、比例、分组）
- 需要用 `pandas` 和 `numpy`

### 必会技能

| 技能 | 代码 | 出现频率 |
|------|------|----------|
| 读取CSV | `pd.read_csv()` | ⭐⭐⭐⭐⭐ |
| 条件分类 | `np.where()` | ⭐⭐⭐⭐⭐ |
| 分箱操作 | `pd.cut()` | ⭐⭐⭐⭐⭐ |
| 分组统计 | `groupby().apply()` | ⭐⭐⭐⭐ |
| 计数 | `value_counts()` | ⭐⭐⭐⭐ |

### ⚠️ 易错点

1. **`pd.cut()` 的 `right=False` 参数**
   ```python
   # right=False 表示左闭右开区间 [0, 18.5)
   pd.cut(data['BMI'], bins=bins, labels=labels, right=False)
   ```

2. **`np.inf` 表示无穷大**
   ```python
   bins = [0, 18.5, 24, 28, np.inf]  # 最后一个区间到无穷大
   ```

3. **`lambda` 函数写法**
   ```python
   data.groupby('分组')['列'].apply(lambda x: (x == '值').mean())
   ```

### 练习题目
- `1.1.1` - 患者住院天数分析（分箱+分组）
- `1.1.2` - 传感器数据统计（分组聚合）
- `1.1.3` - 信用数据分析

---

## 2️⃣ 1.2.X 业务分析文字题

### 题目特点
- 不需要写代码
- 给你一个业务场景描述
- 要求分析问题 + 设计优化方案

### 答题框架

**第一问（问题分析）= 问题描述 + 4个不满原因**

**第二问（优化方案）= 4个实施步骤 + 预期效果**

### ⚠️ 关键记忆点

**问题类型（选2个写）：**
- 准确性不高
- 响应速度慢
- 缺乏个性化
- 系统不稳定

**不满原因（每个问题写4条）：**
- 决策误导
- 资源浪费
- 信任度下降
- 业务损失

**优化步骤（写4个）：**
1. 提高准确性 - 升级算法和技术
2. 加快响应速度 - 优化流程和设备
3. 增加个性化 - 建立用户画像
4. 改善系统稳定性 - 升级架构和监控

### 练习建议
- 背熟 `1.2.X_通用答案模板.md` 文件
- 用自己的话复述一遍
- 考试时套用模板，替换具体场景即可

---

## 3️⃣ 2.1.X 数据清洗预处理题

### 题目特点
- 给你一个有问题的数据文件
- 要求清洗数据（处理缺失值、异常值）
- 需要标准化和划分数据集

### 必会技能

| 技能 | 代码 | 出现频率 |
|------|------|----------|
| 查看数据 | `data.head()` | ⭐⭐⭐⭐⭐ |
| 检查缺失值 | `data.isnull().sum()` | ⭐⭐⭐⭐⭐ |
| 删除缺失值 | `data.dropna()` | ⭐⭐⭐⭐⭐ |
| 类型转换 | `pd.to_numeric()` | ⭐⭐⭐⭐ |
| 标准化 | `StandardScaler()` | ⭐⭐⭐⭐⭐ |
| 划分数据 | `train_test_split()` | ⭐⭐⭐⭐⭐ |
| 保存数据 | `to_csv(index=False)` | ⭐⭐⭐⭐⭐ |

### ⚠️ 易错点

1. **`train_test_split` 参数顺序**
   ```python
   # 返回顺序：X_train, X_test, y_train, y_test
   X_train, X_test, y_train, y_test = train_test_split(
       X, y, test_size=0.2, random_state=42
   )
   ```

2. **`StandardScaler` 用法**
   ```python
   scaler = StandardScaler()
   data[列] = scaler.fit_transform(data[列])  # fit_transform一起用
   ```

3. **保存时不要索引**
   ```python
   data.to_csv('cleaned.csv', index=False)  # index=False很重要
   ```

### 练习题目
- `2.1.1` - 汽车数据清洗
- `2.1.2` - 低碳生活数据处理
- `2.1.3` - 金融数据预处理

---

## 4️⃣ 2.2.X 机器学习建模题

### 题目特点
- 给你清洗好的数据
- 要求训练模型、保存模型、生成预测结果和报告

### 必会技能

| 技能 | 代码 | 出现频率 |
|------|------|----------|
| 逻辑回归 | `LogisticRegression(max_iter=1000)` | ⭐⭐⭐⭐⭐ |
| 随机森林 | `RandomForestRegressor(n_estimators=100)` | ⭐⭐⭐⭐ |
| 训练模型 | `model.fit()` | ⭐⭐⭐⭐⭐ |
| 预测 | `model.predict()` | ⭐⭐⭐⭐⭐ |
| 保存模型 | `pickle.dump()` | ⭐⭐⭐⭐⭐ |
| 分类报告 | `classification_report()` | ⭐⭐⭐⭐⭐ |

### ⚠️ 易错点

1. **分类 vs 回归模型选择**
   - 分类任务（预测类别）→ `LogisticRegression`
   - 回归任务（预测数值）→ `RandomForestRegressor`

2. **pickle保存模型的写法**
   ```python
   # 方法1：with语句（推荐）
   with open('model.pkl', 'wb') as f:
       pickle.dump(model, f)
   
   # 方法2：一行写法
   pickle.dump(model, open('model.pkl', 'wb'))
   ```

3. **分类报告写入文件**
   ```python
   report = classification_report(y_test, y_pred)
   with open('report.txt', 'w') as f:
       f.write(report)
   ```

### 练习题目
- `2.2.1` - 金融违约预测（分类）
- `2.2.2` - 汽车油耗预测（回归）
- `2.2.3` - 健身分析（回归）

---

## 5️⃣ 3.1.X 业务文档分析题

### 题目特点
- 类似1.2.X，是文字题
- 给你一个Excel数据集和业务背景
- 要求分析数据并给出建议

### 答题思路
- 与1.2.X相同，套用模板即可
- 关注数据特点，结合业务场景

---

## 6️⃣ 3.2.X ONNX模型推理题

### 题目特点
- 给你一个预训练的ONNX模型文件
- 给你一张测试图片
- 要求加载模型、预处理图片、进行推理

### 必会技能

| 技能 | 代码 | 出现频率 |
|------|------|----------|
| 加载ONNX | `ort.InferenceSession()` | ⭐⭐⭐⭐⭐ |
| 打开图片 | `Image.open()` | ⭐⭐⭐⭐⭐ |
| 转灰度 | `.convert('L')` | ⭐⭐⭐⭐ |
| 转RGB | `.convert('RGB')` | ⭐⭐⭐⭐ |
| 调整大小 | `.resize()` | ⭐⭐⭐⭐⭐ |
| 转数组 | `np.array()` | ⭐⭐⭐⭐⭐ |
| 添加维度 | `np.expand_dims()` | ⭐⭐⭐⭐⭐ |
| 运行推理 | `session.run()` | ⭐⭐⭐⭐⭐ |
| 取最大值 | `np.argmax()` | ⭐⭐⭐⭐⭐ |

### ⚠️ 易错点

1. **图片模式选择**
   ```python
   # 灰度图（如MNIST手写数字）
   image = Image.open('img.png').convert('L')
   
   # 彩色图（如ResNet图像分类）
   image = Image.open('img.jpg').convert('RGB')
   ```

2. **维度添加顺序**
   ```python
   # 灰度图：先batch，再channel
   img = np.expand_dims(img, axis=0)  # batch维度
   img = np.expand_dims(img, axis=0)  # channel维度 (1,1,H,W)
   
   # 彩色图：需要转置 HWC -> CHW
   img = np.transpose(img, (2, 0, 1))  # (H,W,C) -> (C,H,W)
   img = np.expand_dims(img, axis=0)   # (1,C,H,W)
   ```

3. **推理的输入格式**
   ```python
   input_name = session.get_inputs()[0].name
   output = session.run(None, {input_name: image_array})
   ```

### 练习题目
- `3.2.1` - ResNet图像分类
- `3.2.2` - MNIST手写数字识别
- `3.2.3` - 表情识别
- `3.2.4` - 花卉分类

---

## 🎯 考前一天练习清单

### 上午（2小时）
- [ ] 默写4个代码模板
- [ ] 对照检查，标记错误

### 下午（2小时）
- [ ] 用Jupyter完成1道完整题目（不看答案）
- [ ] 对照答案修正

### 晚上（1小时）
- [ ] 再次默写4个代码模板
- [ ] 背诵1.2.X文字题关键词

---

## 💡 考试技巧

### 时间分配
- 先浏览所有题目，确定题型
- 优先做有把握的题目
- 代码题：先写import，再逐步填写

### 遇到不会的题目
1. 先把能写的写上（import语句、读取数据）
2. 按照模板框架写，能拿部分分
3. 不要留空白

### 代码调试
- 如果报错，检查：
  - 拼写是否正确
  - 引号是否匹配
  - 缩进是否正确
  - 变量名是否一致

---

💪 **相信自己，你已经准备得很充分了！**
