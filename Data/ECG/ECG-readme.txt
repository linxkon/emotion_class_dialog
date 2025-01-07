数据描述

情感对话生成数据集(Emotional Conversation Generation, ECG)包括6000多条句子，分为喜好(Like)，悲伤(Sad)，厌恶(Disgust)，愤怒(Anger)，高兴（Happiness）五类，情绪类别在emotion列给出。 Post为经过人工筛选后的问题，Response为各个模型产生的回复。s1～s4为标注者给出的分数。打分逻辑为：
IF (回复内容合适且通顺)  
　　IF (回复内容正确表达情绪类别)    
　　　分数为2  
　　ELSE   
　　　分数为1
ELSE  
　　分数为0
根据以上打分逻辑进行标注，得到本数据集

数据提供

提供方：清华大学计算机系黄民烈副教授

主页地址：http://coai.cs.tsinghua.edu.cn/hml/

联系方式：aihuang$AT$tsinghua$DOT$edu$DOT$cn

相关论文：Hao Zhou, Minlie Huang, Xiaoyan Zhu, Bing Liu. Emotional Chatting Machine: Emotional Conversation Generation with Internal and External Memory. AAAI 2018, New Orleans, Louisiana, USA.

Github：https://github.com/tuxchow/ecm