# ❓ 常见问题解答 (FAQ)

> 复习过程中遇到问题？这里有答案！

---

## 🐍 Python基础问题

### Q1: `import` 是什么意思？
**A:** `import` 是导入库的意思。Python有很多现成的工具库，`import`就是把这些工具拿过来用。

```python
import pandas as pd    # 导入pandas库，简称pd
import numpy as np     # 导入numpy库，简称np
```

类比：就像用手机下载APP一样，`import`就是"安装"这个工具。

---

### Q2: `pd` 和 `np` 是什么？
**A:** 它们是库的简称（别名）：
- `pd` = `pandas`（数据处理工具）
- `np` = `numpy`（数学计算工具）

这样写代码更短：
```python
# 不用别名（太长）
pandas.read_csv('data.csv')

# 用别名（简洁）
pd.read_csv('data.csv')
```

---

### Q3: 什么是 `DataFrame` ？
**A:** `DataFrame` 就是一个**表格**，有行有列，像Excel表格一样。

```python
data = pd.read_csv('data.csv')  # data就是一个DataFrame
data['列名']                     # 获取某一列
data.head()                      # 查看前5行
```

---

### Q4: `axis=0` 和 `axis=1` 是什么？
**A:** 
- `axis=0` = 按**行**操作（上下方向）
- `axis=1` = 按**列**操作（左右方向）

```python
data.drop('列名', axis=1)  # 删除一列
data.dropna(axis=0)        # 删除有缺失值的行
```

**记忆口诀**：0像一条竖线（行），1像一条横线（列）

---

### Q5: `lambda` 是什么？
**A:** `lambda` 是**一行写完的小函数**。

```python
# 普通函数写法
def is_high_risk(x):
    return (x == '高风险').mean()

# lambda写法（一行搞定）
lambda x: (x == '高风险').mean()
```

考试中只需要背这个固定写法：
```python
data.groupby('分组')['列'].apply(lambda x: (x == '值').mean())
```

---

## 📊 数据处理问题

### Q6: `dropna()` 和 `fillna()` 的区别？
**A:**
- `dropna()` = **删除**有缺失值的行
- `fillna()` = 用某个值**填充**缺失值

```python
data.dropna()              # 删除缺失值所在行
data.fillna(0)             # 用0填充缺失值
data.fillna(method='ffill') # 用前一个值填充
```

考试中一般用 `dropna()` 就够了。

---

### Q7: `pd.cut()` 怎么理解？
**A:** `pd.cut()` 是把连续数值**切成几段**（分箱）。

```python
# 把年龄分成4个区间
bins = [0, 18, 40, 60, 100]     # 分界点
labels = ['少年', '青年', '中年', '老年']  # 标签
data['年龄段'] = pd.cut(data['年龄'], bins=bins, labels=labels)
```

- `right=False` 表示左闭右开区间：`[0, 18)` 包含0，不包含18

---

### Q8: `train_test_split()` 返回的4个值是什么顺序？
**A:** **永远记住这个顺序**：

```python
X_train, X_test, y_train, y_test = train_test_split(X, y, ...)
#  1        2        3        4
```

记忆口诀：**先X后y，先train后test**

---

## 🤖 机器学习问题

### Q9: 什么时候用 LogisticRegression，什么时候用 RandomForestRegressor？
**A:**
- **分类任务**（预测类别，如"是/否"）→ `LogisticRegression`
- **回归任务**（预测数值，如"价格"）→ `RandomForestRegressor`

```python
# 分类：预测是否违约（0或1）
from sklearn.linear_model import LogisticRegression

# 回归：预测汽车油耗（具体数字）
from sklearn.ensemble import RandomForestRegressor
```

---

### Q10: `fit()` 和 `predict()` 是什么？
**A:**
- `fit()` = **训练**模型（让模型学习数据）
- `predict()` = **预测**（用训练好的模型预测新数据）

```python
model.fit(X_train, y_train)    # 用训练数据训练模型
y_pred = model.predict(X_test) # 用测试数据预测
```

---

### Q11: `pickle` 是什么？
**A:** `pickle` 是Python的**模型保存工具**，把训练好的模型保存成文件。

```python
import pickle

# 保存模型
pickle.dump(model, open('model.pkl', 'wb'))

# 加载模型（考试一般不考这个）
model = pickle.load(open('model.pkl', 'rb'))
```

- `'wb'` = write binary（写入二进制）
- `'rb'` = read binary（读取二进制）

---

## 🖼️ ONNX推理问题

### Q12: ONNX是什么？
**A:** ONNX是一种**通用的AI模型格式**，可以在不同平台上运行。考试给你一个`.onnx`文件，你需要用它来识别图片。

---

### Q13: `convert('L')` 和 `convert('RGB')` 的区别？
**A:**
- `convert('L')` = 转成**灰度图**（黑白，1个通道）
- `convert('RGB')` = 转成**彩色图**（红绿蓝，3个通道）

```python
# 手写数字识别（黑白图片）
image = Image.open('digit.png').convert('L')

# 物体分类（彩色图片）
image = Image.open('photo.jpg').convert('RGB')
```

---

### Q14: `np.expand_dims()` 是干什么的？
**A:** **添加一个维度**。神经网络需要特定的输入形状。

```python
# 原始图片形状：(28, 28) - 高度和宽度
img = np.expand_dims(img, axis=0)  # 变成 (1, 28, 28) - 添加batch维度
img = np.expand_dims(img, axis=0)  # 变成 (1, 1, 28, 28) - 添加channel维度
```

---

### Q15: `np.argmax()` 是什么意思？
**A:** 找出数组中**最大值的位置（索引）**。

```python
output = [0.1, 0.05, 0.85]  # 模型输出的概率
result = np.argmax(output)  # 返回 2（因为0.85最大，在第2个位置）
```

在图像分类中，这个索引就对应预测的类别。

---

## 📝 文字题问题

### Q16: 1.2.X题目的"问题"怎么选？
**A:** 从题目描述中找**用户反映最多的两个问题**，一般是：

1. **准确性问题**（识别不准、结果有偏差）
2. **速度问题**（响应慢、等待时间长）
3. **个性化问题**（无法满足特定需求）
4. **稳定性问题**（系统不稳定、容易出错）

---

### Q17: 优化方案写多少内容合适？
**A:** 每个步骤写**2-3句话**即可，不需要太详细：

```
1. 提高准确性
   - 升级算法和技术，提升识别准确率
   - 建立数据校验机制，确保数据质量

2. 加快响应速度
   - 优化系统架构，简化处理流程
   - 升级硬件设备，提高处理能力
```

---

## ⚠️ 常见报错

### E1: `ModuleNotFoundError: No module named 'xxx'`
**A:** 缺少某个库。考试环境应该已经安装好了，如果没有，可以：
```python
!pip install xxx  # 在Jupyter中安装
```

---

### E2: `FileNotFoundError: [Errno 2] No such file or directory`
**A:** 文件路径错误。检查：
1. 文件名是否拼写正确
2. 文件是否和代码在同一目录

---

### E3: `KeyError: '列名'`
**A:** 列名不存在。用 `data.columns` 查看所有列名：
```python
print(data.columns)  # 查看所有列名
```

---

### E4: `ValueError: could not convert string to float`
**A:** 数据类型问题。用 `pd.to_numeric()` 转换：
```python
data['列'] = pd.to_numeric(data['列'], errors='coerce')
data = data.dropna()  # 删除转换失败的行
```

---

## 💡 考试小贴士

### 🔑 代码写不出来怎么办？
1. 先写 `import` 语句（能得分）
2. 再写 `pd.read_csv()` 读取数据（能得分）
3. 按照模板框架填写，即使不完整也能拿部分分

### 🔑 忘记函数名怎么办？
- 写你记得的部分
- 用注释说明你想做什么
  ```python
  # 这里应该用分箱函数把年龄分组
  ```

### 🔑 时间不够怎么办？
- 优先完成有把握的题目
- 代码题拿部分分比空着强
- 文字题按模板快速套用

---

## 📞 还有问题？

复习过程中遇到任何问题，随时可以问我！我会帮你：
- 解释代码含义
- 分析错误原因
- 提供练习建议

**祝你考试顺利！** 💪
