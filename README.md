# learn-llm

## 入门

### 6. 手写GPT2

1. 《Build a Large Language Model (From Scratch)》

Build a Large Language Model (From Scratch) 官方code https://github.com/rasbt/LLMs-from-scratch/tree/main
上述中文版书籍 https://github.com/skindhu/Build-A-Large-Language-Model-CN

可学习知识点：
transform，gpt2架构、文字生成模型、指令微调、分类任务微调、lora微调

学完之后能干嘛:
了解当代深度学习整体面貌和各类工具

### 5. 深度学习入门知识汇总

1. 《动手学深度学习》https://zh.d2l.ai/https://zh.d2l.ai/
代码有官方地址，自行查阅

可学习知识点：
之前学过的深度学习入门系列前3本的汇总+一些新的知识点，整体来说在学习前面3本入门书之后理解上不成大问题了。
也介绍了其他跟并行计算和注意力 transform等有关的知识

学完之后能干嘛:
了解当代深度学习整体面貌和各类工具

### 4. 手写一个简单的深度学习框架
1. 本仓库目录 DLFS3/codes/ 下面有完整随书代码

可学习知识点：
手写：正向计算和反向传播的实现细节、计算图的详细理解，常用的各类神经网络层layer的实现

学完之后能干嘛:
理解现代框架pythorch等基础实现原理

### 3. 手写简单的RNN网络
代码参考：
1. 本仓库目录 DLFS2/codes/ 下面有完整随书代码

可学习知识点：
手写：RNN，简单LSTM, 简单seq2seq，RNN以及对应的复杂版本还是有一点理解成本的，需要多看几遍书和代码，不过现在应该重点学习transform，RNN当做背景快速理解

学完之后能干嘛:
小规模数据集上的文字推理生成

### 2. 手写一个最简单的CNN卷积神经网络

代码参考：
1. 本仓库目录 DeepLearningFromScratch1/codes/ch7 下面有完整随书代码

可学习知识点：
手写：CNN的卷积和池化层的前向和后向传播实现，col2im、im2col实现(难点)

学完之后能干嘛:
识别minst数据集里面的数字图片，测试集上通过率98%+

### 1. 手写一个简单的神经网络(130K参数规模)

代码参考：
1. 本仓库目录 DLFS1/codes下面有完整随书代码
2. 上述书中简要核心代码：demos/DLFS1/simple_neuralne_130K

参考资料：
1. 【深度学习入门：基于python的理由与实现】
2. softmax-with-loss层的梯度反向传播数据推导 https://zhuanlan.zhihu.com/p/86184547  当然书中附录A也给出了计算图的推导过程


可学习知识点：
手写：Affine全连接层(线性变化层、非线性激活函数)、输出层(softmax、softmax-with-loss)、损失函数(均方误差、交叉熵损失)、mini-batch训练方式、数值微分、梯度概念、前向计算梯度、计算图、手动执行梯度反向传播、SGD随机梯度下降(附带Adam、momenttum等)、batch normalization、正则化(dropout、权值衰减等)

学完之后能干嘛:
识别minst数据集里面的数字图片，测试集上通过率93%+

【大模型】

【基础介绍】
1.大语言模型:原理与应用 https://intro-llm.github.io/chapter/LLM-TAP-v2.pdf

【AI应用】
1.【视频风格迁移】


【数学基础】
1. 机器学习的数学 雷明 (https://book.douban.com/subject/35317174/)
2. 线性代数及其应用 (https://book.douban.com/subject/30310517/)
3. 矩阵分析与应用 张贤达
