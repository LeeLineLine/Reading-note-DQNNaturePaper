# Human level control through deep reinforcement learning阅读笔记

------

 - 个人理解

    Agent从高维度的感官输入中有效地表征环境，并通过观察、执行动作和收到反馈与环境完成交互。在执行动作时，确定要执行的动作以获得最大收益，同时，动作的执行将改变环境的状态。因而，对于具体的任务，Agent不仅要学习环境的表示，还要学习决策过程。
    
  利用深度增强学习玩Atari游戏，使用相同的网络结构、超参数值，和学习过程，高得分地通过了49种游戏，实现了Agent的智能化。将4组由图像、动作、得分、下一帧图像组成的序列作为输入，输出对应于游戏的每个动作。采用深度卷积神经网络（DCNN）来学习环境特征，采用experience reply完成了增强学习和DCNN的结合，实现决策的过程。增强学习使用DCNN拟合action-value函数，由Bellman equation采用
\\(Q\left( {s,a;{\theta _i}} \right)\\)来逼近最大期望收益，使用单独的网络$\hat Q$̂更新目标，Q网络用来估计当前的,值$\hat Q$ ̂网络用来估计下一个状态的值，采用随机梯度下降法优化损失函数，后采用ε-greedy策略选定动作。

## 具体内容
### 一、背景
  
 >* 增强学习解决的主要问题：Agent如何优化对环境的控制。
 >- 运用增强学习解决实际问题时，遇到的困难：需要从高维度的感官输入中获得有效的环境表征，并利用其将过去的经验推广到新的情况。
 >- 仿人类：强化学习+分级感官处理
 >- 增强学习之前的应用局限于：可以手工提取特征、低维的领域


### 二、原理

 - 提出新的agent
deep Q-network，使用端到端强化学习，直接从高维度的感官输入中学习到策略。DQN将增强学习和深度神经网络DNN联合起来，其中DNN可以从原始的感观输入数据中提取特征表示。选定了深度卷积神经网络架构，其使用分层的卷积滤波器来模拟感知野的作用。
 - 验证及成果
用DQN玩经典Atari 2600游戏。只需要很少的先验知识，只接收像素和游戏得分作为输入，并且使用相同的算法，网络架构和超参数，DQN架构可以成功地学习一系列不同环境下的控制策略。DQN在49种比赛中达到与专业人类游戏测试者的相当的水平。连接了高维度的感官输入和action，是第一个可扩展学习多任务的agent。
 - 问题描述
Agent通过observation、action和reward的序列与environment（此处为Atari模拟器）完成交互。Agent的目标是通过与模拟器交互，选取最大化future rewards的Actions。
使用深度卷积神经网络来拟合最优的action-value function(动作估值函数)\\(Q^{\ast }\left( s,a\right)\\)，该最优化可以通过observation和action后，执行policy \\(π=P(a|s)\\)来实现，以实现每个时间步t的reward r_t与其衰减系数γ乘积的和的最大化，即：
$\;{Q^*}\left( {s,a} \right) = \mathop {\max }\limits_\pi  E\left[ {{r_t} + \gamma {r_{t + 1}} + {\gamma ^2}{r_{t + 2}} +  \cdots |{s_t} = s,{a_t} = a,\pi } \right]$
> 1）实施难点：增强学习使用神经网络拟合action-value function(Q函数)时很容易不稳定甚至发散。原因：observation序列的相关性；对Q的小的更新会显著影响policy，从而影响数据分布；Q函数和target值% $r + \gamma \mathop {\max }\limits_{a'} Q\left( {s',a'} \right)$的相关性。
>2）解决方法：
   - experience replay,用以打乱数据序列之间的相关性，平滑数据分布的变化。
   - iterative update，仅以target值周期性地更新Q值，减小Q值和target值之间的相关性。

 >3）优点：无需在每次迭代中从头训练网络，更为高效，可成功应用于大型神经网络的训练。
 
 
 使用深度卷积神经网络来近似值函数\\(Q\left( {s,a;{\theta _i}} \right)\\)，其中${\theta _i}$是第i步的Q网络的权重。为实现experience replay，在每一个时间步t，存储agent的experience ${e_t} = \left( {{s_t},{a_t},{r_t},{s_{t + 1}}} \right)$到一个数据集${D_t} = \left\{ {{e_1}, \cdots ,{e_t}} \right\}$中。进行学习时，从数据集中随机抽取样本进行Q-learning更新。在第i步迭代中，Q-learning使用如下的损失函数进行更新：
${L_{i\left( {{\theta _i}} \right)}} = {E_{\left( {s,a,r,s'} \right)~U\left( D \right)}}\left[ {{{\left( {r + \gamma \mathop {\max }\limits_{a'} Q\left( {s',a';{\theta _i}^ - } \right) - Q\left( {s,a;{\theta _i}} \right)} \right)}^2}} \right]$
其中，γ是衰减系数，${\theta _i}$是第i次迭代时Q网络的权重，${\theta _i}^ - $是第i次迭代时用于计算target的网络参数。每隔C步使用Q网络的参数${\theta _i}$来更新{\theta _i}^ - $，其他期间保持不变。
 - 创新点及其神经生物学依据

 >- 采用“端到端”的强化学习，不断地利用reward来塑造CNN结构来表征environment的特征，以促进值估计
 >>依据：知觉学习期间的reward信号可能影响灵长类动物视觉皮层中表征的特征
 
 
 >- experience reply，储存和表征最近的experience，得以结合强化学习和深度神经网络结构
>>依据：海马体与基底神经节交互，支持哺乳动物的大脑通过离线时期的时间压缩，重新激活最近的experience轨迹，来有效更新值函数。
  



### 三、算法细节

 - 数据预处理
>Atari游戏数据，每帧是210×160个像素，128色，进行预处理以降低输入数据的维度。


 >- 对单帧图像进行编码：在已编码的帧和其之前的m帧中，取每个像素颜色值的最大值（此处m=4），以消除图像闪烁



 >- 从RGB中提取Y通道数据（亮度数据），调整为84×84。

算法的ϕ函数：完成数据与处理，将最近的m帧数据堆叠，产生Q函数的输入。

 - 模型结构
>- 使用神经网络的Q参数化方法：之前的方法大多将history和action作为神经网络的输入，但Q函数将history-action对映射为其Q值的标量估计。

 >- 缺点：每个action都需要单独的前向传递来计算Q值，造成与action数量成线性关系的消耗。
>- 不同点：每个action都有一个独立的输出单元（计算Q值），神经网络的输入只有状态表征，输出是输入状态对应的单个action的预测值函数Q。（神经网络的每个输出对应每个动作的预测值函数Q）
>- 优点：在给定状态下，只需对网络进行一次正向传递，就能为所有可能的action计算Q值。


![模型结构][1]

 - 训练细节
>- 在训练中，对于所有的游戏，网络结构、学习算法和超参数的设定保持不变，只改变reward的结构，并且进行分值修剪（将正reward值修剪为+1，负reward值修剪为-1，0值保持不变）。
>- 使用RMSProp优化，minibatch大小为32，策略是ε-greedy算法，在前一百万帧数据中，ε从1线性下降到0.1，之后保持不变。共训练了约5千万帧数据，replay memory的容量是一百万帧。
>>ε-greedy: 就是在当前状态s下，若概率为(1−ε)的，选择在当前估计下最优的action，即${a_t} = argma{x_a}Q\left( {\phi \left( {{s_t}} \right),a;\theta } \right)$，若概率为ε，则随机从剩下的所有actions中选取一个。
>- frame-skipping技术（同别的算法），每隔k帧，agent观察图像并选择action。(k=4)
>- 从Pong, Breakout，Seaquest，Space，Invaders和Beam Rider中非正式搜索确定超参数和优化参数，并应用到其他游戏。
>- 使用以下最小先验知识：输入数据包括视觉图像，特定游戏得分，action的数量和寿命计数。




### [四、算法]
- 理论推导
> 在每个时间步，agent从合法的actions集合$A = \left\{ {1, \cdots ,K} \right\}$中选择一个action${a_t}$。该action被传递到模拟器中，修改其内部状态和游戏得分。

 > Agent观察来自仿真器的图像${x_t} \in {R^d}$，其是当前屏幕像素的表示。Agent接收代表游戏得分变化的reward ${r_t}$。游戏得分可能取决于之前的整个action和observation序列，系统仅在数千个时间步后给出action的反馈。

 >因为agent只能观察当前屏幕，模拟器状态是感知混淆的（不可能仅从当前屏幕x_t完全理解当前情况）。因此，由action和observation序列组成的state，${s_t} = {x_1},{a_1},{x_2}, \cdots ,{a_{t - 1}},{x_t}$，被输入到算法，用以学习游戏策略。

 >假定仿真器中的所有序列都将在有限数量的时间步长内结束，产生了一个大而有限的马尔可夫决策过程（MDP），其中每个序列是一个不同的状态。因此，只需使用完整序列s_t作为时间t处的状态表示，就可以对MDP应用标准强化学习算法。

 >Future rewards是当前时间t到终止时间T的所有reward乘以衰减系数${\gamma ^{t' - t}}$的和。定义为:
${R_t} = \mathop \sum \limits_{t' = t}^T \;{\gamma ^{t' - t}}{r_{t'}}$
其中γ是衰减因子，设置为0.99，T是游戏结束的时间。

 >最佳action-value函数，是在给的状态序列s下，执行action a后，采取任何策略能获得的最大期望收益。其中π是将序列映射为action的策略。
${Q^*}\left( {s,a} \right) = ma{x_\pi }E\left[ {{R_t}|{s_t} = s,{a_t} = a,\pi } \right]$
 
 
  >若在下一个时间步，状态序列$s'$采取所有的action $a'$能取得的最大收益${Q^*}\left( {s',a'} \right)$是已知的，那么最佳策略是选择$a'$以最大化期望值$r + \gamma {Q^*}\left( {s',a'} \right)$。
>>在状态s采取a的最大期望收益等于采取a之后得到的收益r，加上到达状态$s'$后采取最优的$a'$得到最大期望收益的和。

 > 根据Bellman equation，上式可以改写为：
${Q^*}\left( {s,a} \right) = E\left[ {r + \gamma \mathop {\max }\limits_{a'} {Q^*}\left( {s',a'} \right)|s,a} \right]$]
许多强化学习算法背后的基本思想是通过使用Bellman方程作为迭代更新来估计action-value函数，即${Q_{i + 1}}\left( {s,a} \right) = E\left[ {r + \gamma \mathop {\max }\limits_{a'} {Q^*}\left( {s',a'} \right)|s,a} \right]$，当$i \to {\rm{\infty }}$时，${Q_i} \to {Q^*}$。实践中并不可行，转而采用函数$Q\left( {s,a;\theta } \right)$来近似${Q^*}\left( {s,a} \right)$。
 
  
  >引入带有权重θ的卷积神经网络作为Q网络。Q网络可以通过调整第i次迭代中的参数${\theta _i} $来减小Bellman方程中的均方误差。
在每个优化阶段，当优化第i次迭代的损失函数${L_i}\left( {{\theta _i}} \right)$时，我们保持来自先前迭代的参数${\theta _i}^ - $不变，利用${\theta _i}^ - $，得出近似目标值$y = r + \gamma \mathop {\max }\limits_{a'} Q\left( {s',a';{\theta _i}^ - } \right)$)，以代替最优目标值$r + \gamma \mathop {\max }\limits_{a'} {Q^*}\left( {s',a'} \right)$。这导致损失函数序列${L_i}\left( {{\theta _i}} \right)$在每次迭代时都改变。
${L_i}\left( {{\theta _i}} \right) = {E_{s,a,r}}\left[ {{{({E_{s'}}\left[ {y|s,a} \right] - Q\left( {s,a;{\theta _i}} \right))}^2}} \right]$

 > 损失函数对权重的差分如下：
${\nabla _{{\theta _i}}}L\left( {{\theta _i}} \right) = {E_{s,a,r,{s^'}}}\left[ {\left( {r + \gamma \mathop {\max }\limits_{a'} Q\left( {s',a';{\theta _i}^ - } \right) - Q\left( {s,a;{\theta _i}} \right)} \right){\nabla _{{\theta _i}}}Q\left( {s,a;{\theta _i}} \right)} \right]$

  >但实践中通常采用随机梯度下降来优化损失函数，在每个时间步后通过更新权重，用单个样本代替期望，并另${\theta _i}^ -  = {\theta _{i - 1}}$。action通常由ε-greedy策略选定，以(1−ε)的概率选择在当前估计下最优的action，以ε的概率从剩下的所有actions随机选取。


- 训练deep Q-network
> Agent基于Q网络，采用ε-greedy策略来选择和执行actions。
采用$\phi $函数产生Q网络的输入，该算法通过两种方式改进在线Q-learning，使其适用于训练大型神经网络而不发散。
>- experience replay，在每一个时间步t，存储agent的experience ${e_t} = \left( {{s_t},{a_t},{r_t},{s_{t + 1}}} \right)$到一个数据集${D_t} = \left\{ {{e_1}, \cdots ,{e_t}} \right\}$中。进行学习时，从数据集中随机抽取样本进行Q-learning更新。
>>优点：
>>- 一个样本可能被优化的多个迭代过程中被重复采样，提高数据的使用率。
>>-连续采样的样本序列有很强的相关性，随机采样可以打破这种相关性，降低更新的方差。
>>- 避免模型重复做出相同的最优决策，避免局部最优解和不收敛问题。

 >- 使用单独的网络来生成$Q$学习更新中的目标target ${y_j}$。即，每次C更新后，克隆网络$Q$以获得target网络 $\hat Q$，并且使用$\hat Q$来生成$Q$学习的target ${y_j}$，以用于随后的$C$次$Q$更新。
>>- 优点：更新Q时引入延迟，避免发散或振荡，进一步提高神经网络方法的稳定性。

 >- 将误差项的更新$r + \gamma \mathop {\max }\limits_{a'} Q\left( {s',a';{\theta _i}^ - } \right) - Q\left( {s,a;{\theta _i}} \right)$修剪到-1和1之间。
>>- 优点：该误差限制进一步提高了算法的稳定性。




- deep Q-learning with experience replay算法
 > 初始化replay memory D，容量为N；
使用权重$\theta$随机初始化action-value函数Q；
使用权重${\theta _i}^ -  = \theta $初始化target函数$\hat Q$；
在每一个episode
 >> 初始化状态序列${s_1} = {x_1}$，并进行预处理${\phi _1} = \phi \left( {{s_1}} \right)$；
  在每一个时间步t
>>> 根据ε-greedy策略来选择action ${a_t}$
模拟器执行action ${a_t}$，获取reward ${r_t}$和图像${x_{t + 1}}$
更新状态序列${s_{t + 1}} = {s_t},{a_t},{x_{t + 1}}$，进行预处理${\phi _{t + 1}} = \phi \left( {{s_{t + 1}}} \right)$
将序列$\left( {{\phi _t},{a_t},{r_t},{\phi _{t + 1}}} \right)$存储到D中
从D中随机采样minibatch，求解近似目标值${y_j}$
对loss函数执行梯度下降法，更新网络参数
每C步更新$\hat Q = Q$

