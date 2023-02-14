# ChatGpt

Large Language Model(LLM) 

Bert
1. bert 主要做了一个填空，也就是一个预训练任务，预训练任务也就是要根据上下文来判断要填哪个空。
2. 所以 bert 所做的事情就是从大规模的上亿文本语料中，随机的扣掉一部分字，然后形成上面例子的完形填空，不断地学习空格处到底该填写什么。


Gpt 初代
1. openai 早于 bert 出品了一个初代 gpt 模型
2. 大致思想是：基于 transformer 这种编码器，获取文本内部的互相联系
3. gpt 与 bert 的区别是：
    1. bert 仅仅使用 encoder 部分进行模型训练
    2. gpt 仅仅使用 decoder 部分
    3. 两者各自走上了各自的道路，gpt 的 decoder 模型更加适应于文本生成领域

Gpt-2
1. 自从 bert 爆火之后，跟风的模型就更多了，比如 albert，roberta，ernie，bart，xlnet，t5 等等五花八门的模型
2. 最初的子任务只是完形填空就有这么好的效果，于是，为啥不使用其他的语言题型任务呢？
    1. 句子打乱顺序再排序，选择题，判断题，改错题，把预测按单词改成预测实体词汇
3. gpt-2 主要就是在 gpt 的基础上，有添加了多个任务，扩增了数据集和模型参数，又训练了一番

Gpt-3
1. gpt-3 的模型参数和 gpt-2 有非常大的区别，前者的计算量是后者的上千倍。
2. gpt 系列都是采用 decoder 进行训练的，也就是更加适合文本生成的形式，也就是对话模式
3. 训练方法：
    1. 以往的训练都是 2段式的训练方法：首先用大规模的数据集对模型进行预训练，然后再利用下游任务的标注数据集进行 finetune
    2. 但是 gpt-3 使用的是 in-context 的学习方式：
    3. 比如翻译任务
    4. 多个任务同时进行

Chatgpt
1. chatgpt 在模型上和之前的 gpt 没有特别大的变化，但是从训练策略上有所改变
2. 人工操作的 reward：RLHF（Reinforcement Learning from Huma Feedback)
    1. Step1 (collect demonstration data and train a supervised policy)：prompt 会从 prompt dataset 中 sample，人工会给出一些建议，fine-tune gpt-3 with supervised learning
    2. Step2 (collect comparison data and train a reward model): a prompt and several model outputs are sampled, a labeler ranks the outputs from best to worst, this data is used to train our reward model
    3. Step3 (optimize a policy against the reward model using reinforcement learning): a new prompt is sampled from the dataset, the policy generates an output, the reward model calculates a reward for the output, the reward is used to update the policy using ppo
