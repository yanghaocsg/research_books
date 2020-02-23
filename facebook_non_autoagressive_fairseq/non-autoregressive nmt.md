# NON-AUTOREGRESSIVE NEURAL MACHINE TRANSLATION

## Gu jiatao, FB, 2019

## 1 Abstract 

-   Existing approaches to neural machine translation condition each output word on previously generated outputs. We introduce a model that avoids this autoregressive property and produces its outputs in parallel, allowing an order of magnitude lower latency during inference. Through knowledge distillation, the use of input token fertilities as a latent variable, and policy gradient fine-tuning, we achieve this at a cost of as little as 2.0 BLEU points relative to the autoregressive Transformer network used as a teacher. We demonstrate substantial cumulative improvements associated with each of the three aspects of our training strategy, and validate our approach on IWSLT 2016 English–German and two WMT language pairs. By sampling fertilities in parallel at inference time, our non-autoregressive model achieves near-state-of-the-art performance of 29.8 BLEU on WMT 2016 English– Romanian.
-   non-autoregressive +knowledge distillation
-   并行解码，速度快；非自回归，质量好



## 2 Problem

## 2.1 Auto-regressive NMT

-   $X=\{x_1, ..., X_T\},\ Y=\{Y_1, ..., Y_T\} $
-   left-to-right casual structure:
    -   ![1582341581567](1582341581567.png)
    -   $special\ token\ y_0(eg.\ <bos>), y_{T+1}(eg.<eos>)$
-   Maximum Likelihood training : **Cross Entropy**
    -   ![1582341702001](1582341702001.png)
-   Autoregressive NMT without RNN : **Attention**
    -   Transformer

## 2.2 Non-autoregressive Decoding

![1582341818492](1582341818492.png)

![1582341842120](1582341842120.png)

![1582341879980](1582341879980.png)

## 2.3 Multimodality Problem

![1582344208171](1582344208171.png)

## 3 The Non-Autoregressive Transformer (NAT)

## 3.1 Encoder Stack

-   MLP + Multi-head Attention

## 3.2 Decoder Stack

-   Decoder Input
    -   copy source inputs uniformly
    -   copy source inputs using fertilities
-   Non-casual self-attention
-   Positional Attention
-   ![1582344390550](1582344390550.png)

## 3.3 MODELING FERTILITY TO TACKLE THE MULTIMODALITY PROBLEM

![1582344460918](1582344460918.png)

![1582344471293](1582344471293.png)



-   Fertility prediction
-   Benefits of fertility

## 3.4 TRANSLATION PREDICTOR AND THE DECODING PROCESS

-   argmax decoding
    -   ![1582344542459](1582344542459.png)
-   average decoding
    -   ![1582344557843](1582344557843.png)
-   noisy parallel decoding(NPD)
    -   ![1582344582476](1582344582476.png)

## 4 Training

![1582344601881](1582344601881.png)

## 4.1  SEQUENCE-LEVEL KNOWLEDGE DISTILLATION

## 4.2 Fine-tuning

![1582344637684](1582344637684.png)

![1582344654071](1582344654071.png)

## 5 Experiments

![1582344675951](1582344675951.png)

![1582344688288](1582344688288.png)

## 5.2 Results

![1582344709842](1582344709842.png)

-   ablation
-   ![1582344729261](1582344729261.png)
-   ![1582344741098](1582344741098.png)

## 6 Analysis

![1582344802365](1582344802365.png)

![1582344841052](1582344841052.png)

![1582344853793](1582344853793.png)





