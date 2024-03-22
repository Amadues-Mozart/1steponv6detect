# 6Rover: Utilizing Deep Q-Learning for Seed-Free Internet-Wide IPv6 Scanning Target Generation

## 1. 模式提取
### 方法

1. 层次聚类dhc，以前缀为区分，（6forest的 leftmost模式）将数据分裂为共享一个前缀的地址集合，如果分裂后的数量~~小于（x = 100）~~，则停止；✅
2. 基于汉明距离的层次聚类，若距离过大（参数优化），分离（舍弃）汉明距离太远的类  ✅
3. 合并：细粒度的拆分后， 掩盖前缀，基于相似性，把汉明距离相近的模式合并起来，控制一下阈值  （似乎不需要）
4. 模式库：生成以x为前缀时，可能的模式。以种子数为概率（可以调节）
5. 根据模式，生成地址的工具

**问题：**

1 参数和阈值是如何确定的

## 2. 强化学习（上）：EE
臂；下一位的0-f

## 3. 强化学习（下）：DQN
### MDP建模
- State：当前前缀，一个32 位数组，存在部分掩码，有效位为连续的当前前缀
- Action：下一位的Nybble值的选择（0~f），保持当前位，前移一位，共18维
- Reward：根据模式库匹配情况和命中率（为主）共同判定
- DQN：Q(s_t, a_t) = r + /gamma * argmax_a Q(s_t+1, a')

### 训练过程
#### 在n轮的生成过程中进行如下操作：
##### 1.生成训练集
- 创建replay buffer，用以存放<S<sub>t</sub>, a<sub>t</sub>, S<sub>t+1</sub>,r<sub>t</sub>, done>元组，设置batch size为一次训练取用量
- S<sub>t</sub>作为当前状态，作为输入，对于其所有的可能状态，测试得结果计算跳转到每一个状态S<sub>t+1</sub>的初始回报(reward)
- 使用\epsilon-greedy的模式，通过reward来对下一步的状态进行选择，并更新S<sub>t</sub>->S<sub>t+1<sub>
- 将上述行为中产生的<S<sub>t</sub>, a<sub>t</sub>, S<sub>t+1</sub>,r<sub>t</sub>, done>元组存放到replay buffer中待取用
- 当replay buffer容量已满，采取下面策略训练
##### 2.train
- 选取batch size个数据投入到dqn agent中训练
- for tuple in batch:
- &emsp;将state作为dqn的输入，输出为当前状态所有可能action的Q(S<sub>t</sub>, a<sub>t</sub>; w)
- &emsp;从(batch_size, action_dim)的空间中选取对应action的(batch_size,1)向量，作为预测值q<sub>t<sub>
- &emsp;y<sub>t</sub> = r<sub>t</sub> + $\gamma$ * argmax(Q(S<sub>t+1</sub>, a; w)) (使用dqn target网络训练得，可理解为dqn的初始拷贝)
- &emsp;计算loss，更新dqn参数w，w<sub>t+1</sub> = w<sub>t</sub> - $\alpha$ * (q<sub>t</sub> - y<sub>t</sub>) * d<sub>w</sub>
- 经过若干轮后，将dqn target网络更新为dqn网络
- 训练直至收到done = True，该轮结束，开启下一轮生成游戏
### 测试过程
- 对于训练好的dqn网络，设置好初始的state作为输入，根据给出的Q值取argmax作为下一步action的Q值，进行状态跳转，直至生成全部序列
