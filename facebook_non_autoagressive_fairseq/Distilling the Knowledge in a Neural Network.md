# Distilling the Knowledge in a Neural Network

hinton, Google , 2015

## 1 Abstract

-   A very simple way to improve the performance of almost any machine learning algorithm is to train many different models on the same data and then to average their predictions [3]. Unfortunately, making predictions using a whole ensemble of models is cumbersome and may be too computationally expensive to allow deployment to a large number of users, especially if the individual models are large neural nets. Caruana and his collaborators [1] have shown that it is possible to compress the knowledge in an ensemble into a single model which is much easier to deploy and we develop this approach further using a different compression technique. We achieve some surprising results on MNIST and we show that we can significantly improve the acoustic model of a heavily used commercial system by distilling the knowledge in an ensemble of models into a single model. We also introduce a new type of ensemble composed of one or more full models and many specialist models which learn to distinguish fine-grained classes that the full models confuse. Unlike a mixture of experts, these specialist models can be trained rapidly and in parallel.
-   ensemble 能取得最好效果， 但部署和推理性能跟不上
-   提出了一个方法，将多个模型知识压缩到一个模型，容易部署
-   介绍了一种ensemble的新模型，集成一批模型或者专家系统，而且可以快速训练和并行训练

## 2 

