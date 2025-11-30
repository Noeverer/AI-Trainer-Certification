# 🎯 人工智能训练师三级 - 代码速记卡

> 📱 专为手机阅读优化 | ⏰ 考前必背

---

## ⚠️ 重要提示

**考试时 `import` 语句已经写好，不需要你写！**

你只需要专注于**填写核心代码逻辑**即可。

---

## 🔥 模板1: 数据分析统计 (1.1.X)

### 完整代码模板

```python
import pandas as pd
import numpy as np

# 1️⃣ 读取数据
data = pd.read_csv('xxx.csv')

# 2️⃣ 条件分类（用np.where）
data['风险等级'] = np.where(
    data['住院天数'] > 7, 
    '高风险', 
    '低风险'
)

# 3️⃣ 统计数量和占比
counts = data['风险等级'].value_counts()
ratio = counts / len(data)
print(counts)
print(ratio)

# 4️⃣ 分箱操作（用pd.cut）
bins = [0, 18.5, 24, 28, np.inf]
labels = ['偏瘦', '正常', '超重', '肥胖']
data['BMI分组'] = pd.cut(
    data['BMI'], 
    bins=bins, 
    labels=labels, 
    right=False
)

# 5️⃣ 分组统计比例
比例 = data.groupby('BMI分组')['风险等级'].apply(
    lambda x: (x == '高风险').mean()
)

# 6️⃣ 分组统计数量
数量 = data['BMI分组'].value_counts()
```

### 🔑 关键点速记

| 功能 | 代码 | 记忆口诀 |
|------|------|----------|
| 读CSV | `pd.read_csv('文件.csv')` | pd读csv |
| 条件判断 | `np.where(条件, 是, 否)` | np问哪里 |
| 分箱 | `pd.cut(列, bins, labels)` | pd切分 |
| 分组统计 | `groupby('列').apply()` | 按组应用 |
| 计数 | `value_counts()` | 值计数 |

---

## 🔥 模板2: 数据清洗预处理 (2.1.X)

### 完整代码模板

```python
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# 1️⃣ 读取数据
data = pd.read_csv('xxx.csv')
# 或读Excel
data = pd.read_excel('xxx.xlsx')

# 2️⃣ 查看前5行
print(data.head())

# 3️⃣ 检查缺失值
print(data.isnull().sum())

# 4️⃣ 删除缺失值
data = data.dropna()

# 5️⃣ 转换数据类型（处理异常值）
data['列名'] = pd.to_numeric(
    data['列名'], 
    errors='coerce'
)
data = data.dropna()

# 6️⃣ 标准化数值列
scaler = StandardScaler()
数值列 = ['列1', '列2', '列3']
data[数值列] = scaler.fit_transform(data[数值列])

# 7️⃣ 选择特征和目标
X = data[['特征1', '特征2', '特征3']]
y = data['目标列']

# 8️⃣ 划分训练集测试集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2, 
    random_state=42
)

# 9️⃣ 保存清洗后的数据
data.to_csv('cleaned_data.csv', index=False)
```

### 🔑 关键点速记

| 功能 | 代码 | 记忆口诀 |
|------|------|----------|
| 删缺失值 | `data.dropna()` | drop掉NA |
| 检查缺失 | `data.isnull().sum()` | 是空求和 |
| 类型转换 | `pd.to_numeric(列, errors='coerce')` | 转数字强制 |
| 标准化 | `scaler.fit_transform(列)` | 拟合转换 |
| 划分数据 | `train_test_split(X,y,test_size=0.2)` | 训练测试分 |

---

## 🔥 模板3: 机器学习建模 (2.2.X)

### 分类任务模板（LogisticRegression）

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import pickle

# 1️⃣ 加载数据
data = pd.read_csv('xxx.csv')

# 2️⃣ 选择特征和目标
X = data.drop(['目标列'], axis=1)
y = data['目标列']

# 3️⃣ 划分数据集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2, 
    random_state=42
)

# 4️⃣ 创建并训练模型
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# 5️⃣ 保存模型
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

# 6️⃣ 预测
y_pred = model.predict(X_test)

# 7️⃣ 保存预测结果
pd.DataFrame(y_pred, columns=['预测结果']).to_csv(
    'results.txt', index=False
)

# 8️⃣ 生成评估报告
report = classification_report(y_test, y_pred)
with open('report.txt', 'w') as f:
    f.write(report)

# 9️⃣ 计算准确率
accuracy = (y_test == y_pred).mean()
print(f"准确率: {accuracy:.2f}")
```

### 回归任务模板（RandomForest）

```python
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import pickle

# 创建随机森林回归模型
model = RandomForestRegressor(
    n_estimators=100,  # 100棵树
    random_state=42
)
model.fit(X_train, y_train)

# 预测
y_pred = model.predict(X_test)

# 评估指标
mse = mean_squared_error(y_test, y_pred)  # 均方误差
r2 = r2_score(y_test, y_pred)  # R²分数
train_score = model.score(X_train, y_train)
test_score = model.score(X_test, y_test)

# 保存报告
with open('report.txt', 'w') as f:
    f.write(f'训练集得分: {train_score}\n')
    f.write(f'测试集得分: {test_score}\n')
    f.write(f'均方误差: {mse}\n')
    f.write(f'R²分数: {r2}\n')
```

### 🔑 关键点速记

| 功能 | 代码 | 记忆口诀 |
|------|------|----------|
| 逻辑回归 | `LogisticRegression(max_iter=1000)` | 逻辑1000次 |
| 随机森林 | `RandomForestRegressor(n_estimators=100)` | 森林100棵 |
| 训练 | `model.fit(X_train, y_train)` | 模型拟合 |
| 预测 | `model.predict(X_test)` | 模型预测 |
| 保存模型 | `pickle.dump(model, open('x.pkl','wb'))` | 泡菜存模型 |
| 分类报告 | `classification_report(y_test, y_pred)` | 分类报告 |

---

## 🔥 模板4: ONNX模型推理 (3.2.X)

### 简单图像分类（如MNIST手写数字）

```python
import onnxruntime as ort
import numpy as np
from PIL import Image

# 1️⃣ 加载ONNX模型
session = ort.InferenceSession('model.onnx')

# 2️⃣ 加载图片
image = Image.open('img.png').convert('L')  # L=灰度图

# 3️⃣ 预处理图片
image = image.resize((28, 28))  # 调整大小
image_array = np.array(image).astype(np.float32)

# 4️⃣ 添加维度 (batch, channel)
image_array = np.expand_dims(image_array, axis=0)  # batch维度
image_array = np.expand_dims(image_array, axis=0)  # channel维度

# 5️⃣ 获取输入名称并推理
input_name = session.get_inputs()[0].name
output = session.run(None, {input_name: image_array})

# 6️⃣ 获取预测结果
predicted = np.argmax(output[0])
print(f"预测结果: {predicted}")
```

### 复杂图像分类（如ResNet）

```python
import onnxruntime as ort
import numpy as np
import scipy.special
from PIL import Image

# 预处理函数
def preprocess_image(image, size=224):
    image = image.resize((size, size))
    image = np.array(image).astype(np.float32)
    image = image / 255.0  # 归一化
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    image = (image - mean) / std
    image = np.transpose(image, (2, 0, 1))  # HWC -> CHW
    image = np.expand_dims(image, axis=0)  # 添加batch维度
    return image.astype(np.float32)

# 加载模型和标签
session = ort.InferenceSession('resnet.onnx')
with open('labels.txt') as f:
    labels = [line.strip() for line in f.readlines()]

# 加载并处理图片
image = Image.open('img.jpg').convert('RGB')
processed = preprocess_image(image)

# 推理
input_name = session.get_inputs()[0].name
output = session.run(None, {input_name: processed})[0]

# 应用softmax获取概率
probs = scipy.special.softmax(output, axis=-1)

# 获取Top-5结果
top5_idx = np.argsort(probs[0])[-5:][::-1]
for i, idx in enumerate(top5_idx):
    print(f"{i+1}: {labels[idx]} - {probs[0][idx]:.2%}")
```

### 🔑 关键点速记

| 功能 | 代码 | 记忆口诀 |
|------|------|----------|
| 加载模型 | `ort.InferenceSession('x.onnx')` | ort会话 |
| 灰度图 | `Image.open(x).convert('L')` | 转L灰度 |
| RGB图 | `Image.open(x).convert('RGB')` | 转RGB彩色 |
| 调整大小 | `image.resize((28, 28))` | resize尺寸 |
| 添加维度 | `np.expand_dims(arr, axis=0)` | 扩展维度 |
| 获取输入名 | `session.get_inputs()[0].name` | 获取输入名 |
| 推理 | `session.run(None, {name: data})` | 运行推理 |
| 取最大值索引 | `np.argmax(output)` | 取最大下标 |

---

## 📝 1.2.X 文字题答题模板

### 第一问：问题分析

```
问题一：【XX准确性/识别准确性不高】

问题描述：
系统的[功能]准确率较低，经常出现[具体错误]，
影响[业务目标]的实现。

用户不满原因：
1. 决策误导：不准确的结果导致用户做出错误决策
2. 资源浪费：基于错误结果的投入无法产生预期效果
3. 信任度下降：用户对系统可靠性产生怀疑
4. 业务损失：无法及时发现和处理问题

---

问题二：【响应速度慢/处理时间长】

问题描述：
系统处理[任务]的时间过长，用户需要等待较长时间
才能获得结果，影响实时决策。

用户不满原因：
1. 效率低下：无法快速响应变化，错过最佳处理时机
2. 用户体验差：长时间等待降低使用意愿
3. 竞争劣势：相比竞争对手响应能力处于不利地位
4. 业务延误：无法及时处理紧急情况
```

### 第二问：优化方案

```
优化方案概述：
针对[系统名称]存在的问题，设计相应的优化方案。

关键实施步骤：

1️⃣ 提高准确性
- 升级算法和技术，提升识别/处理准确性
- 建立数据校验机制，确保数据质量

2️⃣ 加快响应速度
- 优化系统架构，简化处理流程
- 升级硬件设备，提高处理能力

3️⃣ 增加个性化功能
- 建立用户画像模型，了解用户需求
- 提供个性化服务和定制化选项

4️⃣ 改善系统稳定性
- 建立系统监控机制，实时监控状态
- 建立自动故障恢复机制

预期优化效果：
- 准确率显著提升
- 响应速度大幅改善
- 个性化服务能力增强
- 系统稳定性明显提高
```

---

## ⚡ 考场急救速记

### 万能开头
```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
```

### 读数据
```python
data = pd.read_csv('xxx.csv')      # CSV文件
data = pd.read_excel('xxx.xlsx')   # Excel文件
```

### 数据清洗三连
```python
data = data.dropna()               # 删缺失
data = data.drop_duplicates()      # 删重复
data.to_csv('clean.csv', index=False)  # 保存
```

### 机器学习三连
```python
model.fit(X_train, y_train)        # 训练
y_pred = model.predict(X_test)     # 预测
pickle.dump(model, open('m.pkl','wb'))  # 保存
```

### ONNX推理三连
```python
session = ort.InferenceSession('x.onnx')  # 加载
output = session.run(None, {name: data})  # 推理
result = np.argmax(output[0])             # 结果
```

---

## 🎯 考前最后检查

### ✅ 确认你能默写

- [ ] `import pandas as pd`
- [ ] `import numpy as np`
- [ ] `from sklearn.model_selection import train_test_split`
- [ ] `pd.read_csv('xxx.csv')`
- [ ] `data.dropna()`
- [ ] `np.where(条件, 是, 否)`
- [ ] `pd.cut(列, bins, labels)`
- [ ] `train_test_split(X, y, test_size=0.2, random_state=42)`
- [ ] `model.fit(X_train, y_train)`
- [ ] `model.predict(X_test)`
- [ ] `pickle.dump(model, open('x.pkl', 'wb'))`
- [ ] `ort.InferenceSession('x.onnx')`
- [ ] `np.argmax(output)`

### ✅ 1.2.X文字题关键词

**问题类型**：准确性、响应速度、个性化、稳定性

**不满原因**：决策误导、资源浪费、信任度下降、业务损失

**优化方向**：升级算法、优化流程、建立机制、改善体验

---

💪 **加油！你一定能过的！**
