# LLM记忆流与反思机制 - 技术文档

## 概述

本文档简单实现论文《Generative Agents: Interactive Simulacra of Human Behavior》中描述的记忆流(Memory Stream)和反思(Reflection)机制。该系统为LLM agent 提供了长期记忆和高层次推理能力。

## 核心类定义

### 1. Memory 类 - 记忆对象

记忆是系统中的基本数据单元，代表agent的一次观察、思考或行为。

#### 构造函数
```python
def __init__(self, description, creation_timestamp, most_recent_access_timestamp, memory_type)
```

#### 属性定义
- `self.description`: str - 记忆内容描述
- `self.creation_timestamp`: datetime - 记忆创建的时间戳
- `self.most_recent_access_timestamp`: datetime - 最近访问的时间戳
- `self.importance`: float - 重要性得分（通过generate_importance_score方法计算）
- `self.embedding_vector`: np.array - 记忆的向量嵌入（通过generate_embedding_vector方法计算）
- `self.memory_type`: str - 记忆类型（"O"=observation观察, "R"=reflection反思, "P"=plan计划）

#### 需要实现的方法

**`generate_embedding_vector(self, input_text) -> np.array`**
- 功能：为输入文本生成向量嵌入
- 参数：input_text: str - 需要生成嵌入的文本
- 实现：使用开源的embedding模型(huggingface上搜索)或者OpenAI的text-embedding模型生成嵌入
- 返回：np.array - 向量嵌入

**`generate_importance_score(self, input_text) -> float`**
- 功能：使用LLM评估记忆内容的重要性得分
- 参数：
  - input_text: str - 记忆内容
- 实现：调用LLM如GPT评估记忆内容的重要性
- 返回：float - 0.1到1.0之间的重要性得分（原始1-10分除以10）

### 2. MemoryStream 类 - 记忆流管理器

记忆流是管理所有记忆的类，负责存储、检索和组织记忆。

#### 构造函数
```python
def __init__(self, memories, alpha_recency, alpha_importance, alpha_relevance, decay_factor, context_window_size)
```

#### 属性定义
- `self.memories`: List[Memory] - 所有记忆对象的列表
- `self.alpha_recency`: float - 新近性得分权重
- `self.alpha_importance`: float - 重要性得分权重  
- `self.alpha_relevance`: float - 相关性得分权重
- `self.decay_factor`: float - 时间衰减因子（如0.99）
- `self.context_window_size`: int - top-k的参数

#### 需要实现的方法

**`add_memory(self, memory) -> None`**
- 功能：添加新记忆到记忆流
- 参数：memory: Memory - 要添加的记忆对象
- 实现：直接将记忆对象添加到memories列表中
- 返回：无

**`retrieve_memories(self, query_memory, filter_statement=None) -> pd.DataFrame`**
- 功能：基于查询记忆检索最相关的记忆
- 参数：
  - query_memory: Memory - 查询记忆对象
  - filter_statement: str - 可选的过滤条件（如'relevance > 0.5'）
- 实现：
  1. 计算所有记忆的新近性得分（使用decay_factor的时间衰减）
  2. 计算所有记忆与查询的相关性得分（使用余弦相似度）
  3. 归一化新近性和相关性得分
  4. 计算综合得分：alpha_recency * 新近性 + alpha_importance * 重要性 + alpha_relevance * 相关性
  5. 应用可选过滤条件
  6. 按得分排序，返回top-k记忆
  7. 更新检索到记忆的most_recent_access_timestamp
- 返回：pd.DataFrame - 包含description, importance, recency, relevance, pointer, score列

**`get_memory_df(self) -> pd.DataFrame`**
- 功能：获取所有记忆的DataFrame表示
- 参数：无
- 实现：将所有记忆转换为包含所有属性的DataFrame
- 返回：pd.DataFrame - 包含description, importance, creation_timestamp, most_recent_access_timestamp, memory_type, embedding_vector列

### 3. Reflection 类 - 反思记忆（继承自Memory）

反思是一种特殊类型的记忆，包含对其他记忆的高层次洞察。

#### 构造函数
```python
def __init__(self, description, creation_timestamp, most_recent_access_timestamp, pointers)
```

#### 属性定义
- 继承Memory的所有属性
- `self.pointers`: List[int] - 指向作为证据的记忆索引列表
- `self.memory_type`: str - 自动设置为"reflection"

## 反思机制相关函数

### 1. parse_insights函数
```python
def parse_insights(insight_responses) -> List[Tuple[str, List[int]]]
```
- 功能：解析LLM生成的洞察响应，提取洞察内容和证据索引
- 参数：insight_responses: str - LLM生成的洞察文本
- 实现：使用正则表达式匹配格式"数字. 洞察内容 [索引列表]"
- 返回：List[Tuple[str, List[int]]] - 洞察文本和对应证据索引的元组列表

#### parse_insight输入输出的例子
```python
# 输入例子
insight_responses = """1. Missing data in user_dims table for country may impact accuracy of data analysis and decision-making for marketing campaigns and user segmentation. [7, 52, 47]
2. Inconsistent information in user_dims table may lead to incorrect analysis and decision-making for user segmentation and marketing campaigns. [22, 40]
3. Duplicated and inconsistent data in user_dims table may impact accuracy of data-driven decisions for user segmentation and marketing campaigns. [3, 57, 62, 42]
4. Large number of missing values in bitcoin_price_data table may affect accuracy of analysis and decision-making for cryptocurrency investments. [14]"""
#输出例子
result = [
    (
        "Missing data in user_dims table for country may impact accuracy of data analysis and decision-making for marketing campaigns and user segmentation.",
        [7, 52, 47]
    ),
    (
        "Inconsistent information in user_dims table may lead to incorrect analysis and decision-making for user segmentation and marketing campaigns.",
        [22, 40]
    ),
    (
        "Duplicated and inconsistent data in user_dims table may impact accuracy of data-driven decisions for user segmentation and marketing campaigns.",
        [3, 57, 62, 42]
    ),
    (
        "Large number of missing values in bitcoin_price_data table may affect accuracy of analysis and decision-making for cryptocurrency investments.",
        [14]
    )
]
```

### 2. condense_insights函数
```python
def condense_insights(insight_list) -> str
```
- 功能：将多个洞察浓缩为最多5个要点
- 参数：insight_list: str - 原始洞察列表文本
- 实现：调用LLM如GPT进行内容浓缩
- 返回：str - 浓缩后的洞察文本

#### condense_insights输入输出的例子
```python
#输入例子
insight_list = """1. The user_dims table appears to have significant data quality issues affecting country information which could impact marketing campaigns [7, 52, 47]
2. There are inconsistencies in the user_dims table that might lead to problems in user segmentation [22, 40]  
3. The user_dims table has duplicate entries for some users which is problematic [3, 57]
4. Missing country data in user_dims affects geographical analysis [62, 42]
5. The bitcoin_price_data table has a large number of missing values which is concerning for analysis [14]
6. Data quality issues in user_dims extend beyond just country information [26, 7]
7. User segmentation accuracy is compromised due to data inconsistencies [22, 12]
8. Marketing campaign effectiveness may be reduced due to missing geographical data [47, 52]"""
#参考system_prompt和prompt
system_prompt = '''
You are an AI who is an expert at condensing information into concise numbered, bullet points. 
Please condense the following insights into a maximum of 5 bullet points, each starting with a number followed by a period.
Include all the appropriate index numbers referencing the information in square brackets, listed at the end of each insight.
Do not include multiple, unrelated insights in one bullet point. Ensure each insight explicitly mentions the subject by name. 
'''

prompt = f"""
Please condense the insights generated in this list:

{insight_list}

"""
#输出例子
condensed_output = """1. Missing data in user_dims table for country may impact accuracy of data analysis and decision-making for marketing campaigns and user segmentation. [7, 52, 47]
2. Inconsistent information in user_dims table may lead to incorrect analysis and decision-making for user segmentation and marketing campaigns. [22, 40]
3. Duplicated and inconsistent data in user_dims table may impact accuracy of data-driven decisions for user segmentation and marketing campaigns. [3, 57, 62, 42]
4. Large number of missing values in bitcoin_price_data table may affect accuracy of analysis and decision-making for cryptocurrency investments. [14]
5. The user_dims table has data issues for many users. [26, 7, 22, 12, 17, 62, 47, 57, 52]"""
```

### 3. generate_insight函数
```python
def generate_insight(memory_stream, context_window, threshold) -> Tuple[List[str], List[List[Tuple[str, List[int]]]]]
```
- 功能：生成反思洞察的核心函数
- 参数：
  - memory_stream: MemoryStream - 记忆流对象
  - context_window: int - 用于反思的最近记忆数量
  - threshold: float - 触发反思的重要性阈值

- 实现：
  1. 获取最近的context_window个记忆
  2. 检查重要性总和是否达到threshold
  3. 生成3个高层次问题
  4. 为每个问题检索相关记忆（relevance > 0.5）
  5. 为每个问题(基于检索到的记忆)生成洞察
  6. 浓缩洞察
- 返回：Tuple[List[str], List[List[Tuple[str, List[int]]]]] - 问题列表和洞察列表

#### generate_insight输出的例子
```python
# 调用generate_insight得到
(questions_fmt, insights_list)

questions_fmt = [
    "How does the large number of missing values for country in the user_dims table impact data analysis?",
    "How can the missing data for some user IDs in the wallet_event_logs table affect user behavior analysis?", 
    "What impact does the large number of outliers in bitcoin price data have on trend analysis?"
]
insights_list = [
    # 第一个问题对应的洞察列表
    [
        ("Missing data in user_dims table for country may impact accuracy of data analysis", [7, 52, 47]),
        ("Inconsistent information in user_dims table may lead to incorrect analysis", [22, 40]),
        ("Duplicated data in user_dims table may impact data-driven decisions", [3, 57, 62])
    ],
    # 第二个问题对应的洞察列表  
    [
        ("The user_dims table has data issues for many users", [26, 7, 22, 12]),
        ("Missing user IDs in wallet_event_logs affects behavior analysis", [53, 63, 43])
    ],
    # 第三个问题对应的洞察列表
    [
        ("Bitcoin price data has missing and inconsistent data affecting analysis", [9, 14, 44, 54]),
        ("Bitcoin transactions have missing and duplicated data", [10, 15, 20, 29]),
        ("Daily bitcoin price outliers impact historical trend analysis", [54])
    ]
]
```

### 4. generate_reflection函数
```python
def generate_reflection(memory_stream, insights_list) -> MemoryStream
```
- 功能：将生成的洞察转换为Reflection对象并添加到记忆流
- 参数：
  - memory_stream: MemoryStream - 记忆流对象
  - insights_list: List[List[Tuple[str, List[int]]]] - 洞察列表
- 实现：遍历所有洞察，为每个洞察创建Reflection对象并添加到记忆流
- 返回：MemoryStream - 更新后的记忆流对象

## 核心Prompt模板

### 1. 重要性评估Prompt

```python
system_intel = '''
You are an AI who rates events in terms of their perceived importance on a scale of 1 to 10.
'''

context = f"""
On the scale of 1 to 10, where 1 is purely mundane
(e.g., brushing teeth, making bed) and 10 is
extremely poignant (e.g., a break up, college
acceptance), rate the likely poignancy of the
following piece of memory.
Memory: {input_text}
Rating: <fill in with a number between 1 and 10>
Please return only the number rating as an integer (e.g. 5)
Do not explain why you gave this rating.
"""
```

### 2. 反思问题生成Prompt

```python
questions_intel = '''
You are an intelligent AI system designed to reflect on past experiences and generate meaningful insights. 
Your goal is to help the user gain a deeper understanding of their experiences and identify patterns or themes that may not have been immediately obvious. 
When generating questions, focus on high-level, abstract questions that will prompt the subject to think more deeply about their experiences. 
Avoid simple, surface-level questions that can easily be answered by the memories themselves. 
'''

question_context = f"""
Given only the information above, what are 3 most salient high-level questions we can answer about the subjects in the statements? 
Be as specific as possible and refer to subjects explicitly by their name.

Statements about subjects
{recent_memory_prompt}"""
```

### 3. 洞察生成Prompt

```python
insight_intel = '''
You are an intelligent AI system designed to reflect on past experiences and generate meaningful insights.
Your goal is to help yourself gain a deeper understanding of your experiences and identify patterns or themes that may not have been immediately obvious.
When generating insights, focus on high-level, abstract insights that will prompt you to think more deeply about your experiences.
Cite the index of the statement as evidence, do not explicitly mention the statement or evidence. The desired format is: "
1. insight [`index number(s) of evidence`]
2. insight [`index number(s) of evidence`]
3. insight [`index number(s) of evidence`]
etc"

REQUIREMENTS:
- You MUST ensure to use square brackets enclosing the index numbers at the end of each insight.
- You MUST only include one insight per line.
- You MUST be as concise and specific as possible
'''

insight_context = f"What high-level insights can you infer from the above statements? Cite the index of the statement as evidence, do not explicitly mention the statement or evidence. Be as specific and concise as possible; mention explicitly subject names for each insight. (example format: - insight [`index of evidence`] 

Statements about subjects
{relevant_memories_for_question.description}

{question}"
```

### 4. 洞察浓缩Prompt

```python
intel = '''
You are an AI who is an expert at condensing information into concise numbered, bullet points. 
Please condense the following insights into a maximum of 5 bullet points, each starting with a number followed by a period.
Include all the appropriate index numbers referencing the information in square brackets, listed at the end of each insight.
Do not include multiple, unrelated insights in one bullet point. Ensure each insight explicitly mentions the subject by name. 
'''

context = f"""
Please condense the insights generated in this list:

{insight_list}

"""
```

## 核心算法实现

### 1. 记忆检索得分计算
```python
# 新近性得分计算（时间衰减）
elapsed_time = (now - memory.most_recent_access_timestamp).total_seconds()
recency_score = np.power(decay_factor, elapsed_time)

# 相关性得分计算（余弦相似度）
relevance_score = 1 - cosine(query_memory.embedding_vector, memory.embedding_vector)

# 得分归一化
recency_normalized = (recency - min) / (max - min)
relevance_normalized = (relevance - min) / (max - min)

# 综合得分计算
total_score = (alpha_recency * recency_normalized + 
               alpha_importance * importance + 
               alpha_relevance * relevance_normalized)
```

### 2. 反思触发逻辑
```python
# 检查最近记忆的重要性总和
recent_memories = memory_stream.memories[-context_window:]
recent_memory_importance = [memory.importance for memory in recent_memories]
if sum(recent_memory_importance) < threshold:
    return None  # 不触发反思
```

### 3. 重要性得分处理
```python
# 从LLM响应中提取数字评分
score = float(re.sub(r'[^\d]', '', responses.choices[0]['message']["content"]))
if math.isnan(score) or (score/10.0 > 10 or score/10.0 < 0):
    raise ValueError("Score is NaN")
return score/10.0  # 转换为0.1-1.0范围
```

## 使用流程示例

### 1. 初始化记忆流
```python
memory_stream = MemoryStream(
    memories=[],           # 空的记忆列表
    alpha_recency=1,       # 新近性权重
    alpha_importance=1,    # 重要性权重
    alpha_relevance=1,     # 相关性权重
    decay_factor=0.99,     # 时间衰减因子
    context_window_size=5  # 上下文窗口大小
)
```

### 2. 添加记忆
```python
for memory_desc in memory_list:
    memory = Memory(
        description=memory_desc,
        creation_timestamp=datetime.now(),
        most_recent_access_timestamp=datetime.now(),
        memory_type='observation'
    )
    memory_stream.add_memory(memory)
```

### 3. 检索记忆
```python
query_memory = Memory(
    description="查询内容",
    creation_timestamp=datetime.now(),
    most_recent_access_timestamp=datetime.now(),
    memory_type="observation"
)
retrieved_memories = memory_stream.retrieve_memories(query_memory)
```

### 4. 生成反思
```python
questions, insights_list = generate_insight(
    memory_stream=memory_stream,
    context_window=15,  # 使用最近15个记忆
    threshold=3         # 重要性阈值
)
memory_stream = generate_reflection(memory_stream, insights_list)
```


## 测试用例设计(so on)

```python
def test_memory_creation():
    memory = Memory("test description", datetime.now(), datetime.now(), "observation")
    assert memory.description == "test description"
    assert memory.memory_type == "observation"
    assert 0 <= memory.importance <= 1

def test_reflection_creation():
    reflection = Reflection("test insight", datetime.now(), datetime.now(), [1, 2, 3])
    assert reflection.memory_type == "reflection" 
    assert reflection.pointers == [1, 2, 3]

def test_memory_retrieval():
    # 测试检索算法的正确性
    pass
```
