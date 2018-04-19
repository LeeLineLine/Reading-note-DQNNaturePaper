# Reading-note-DQNNaturePaper
Human level control through deep reinforcement learning阅读笔记

个人理解：
Agent从高维度的感官输入中有效地表征环境，并通过观察、执行动作和收到反馈与环境完成交互。在执行动作时，确定要执行的动作以获得最大收益，同时，动作的执行将改变环境的状态。因而，对于具体的任务，Agent不仅要学习环境的表示，还要学习决策过程。
利用深度增强学习玩Atari游戏，使用相同的网络结构、超参数值，和学习过程，高得分地通过了49种游戏，实现了Agent的智能化。
将4组由图像、动作、得分、下一帧图像组成的序列作为输入，输出对应于游戏的每个动作。采用深度卷积神经网络（DCNN）来学习环境特征，采用experience reply完成了增强学习和DCNN的结合，实现决策的过程。增强学习使用DCNN拟合action-value函数，由Bellman equation采用Q(s,a;θ_i )来逼近最大期望收益，使用单独的网络Q ̂更新目标，Q网络用来估计当前的值，Q ̂网络用来估计下一个状态的值，采用随机梯度下降法优化损失函数，后采用ε-greedy策略选定动作。
