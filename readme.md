# RNN from Scratch 

![generated with d-id.com](imgs/frankensteins_robot.png)

A custom implementation of RNN's architecture from scratch with pytorch. The RNN with 2 hidden layers was trained on Mary Shelley's Frankenstein for predicting next token. The performance of the RNN was compared to a LSTM network.

![](imgs/train_loss.png)
![](imgs/val_loss.png)

Although the implemetation seems to work, it's also clear that the custom network gets overfitted very fast, so the default architecture isn't powerful enough to learn any pattern from the text. 
