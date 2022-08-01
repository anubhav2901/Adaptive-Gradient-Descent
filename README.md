# Adaptive-Gradient-Descent
In this repository, I implement the Adaptive Gradient Descent from scratch using Python. 
Adaptive gradient descent algorithm is introduced in the work: *"Yura Malitsky, Konstantin Mishchenko Proceedings of the 37th International Conference on Machine Learning, PMLR 119:6702-6712, 2020."*

This work can be extended by integrating other methods such as Flooding loss introduced by: *"Takashi Ishida, Ikko Yamane, Tomoya Sakai, Gang Niu, and Masashi Sugiyama. 2020. Do we need zero training loss after achieving zero training error? In Proceedings of the 37th International Conference on Machine Learning (ICML'20). JMLR.org, Article 428, 4604–4614""*

## Algorithm
![algo](https://github.com/anubhav2901/Adaptive-Gradient-Descent/blob/main/figures/AdaptiveGD.png)

## Results
Following figures compares the losses and accuracies for the Logistic regression classifier trained using Stochastic Gradient descent and Adaptive Gradient descent algorithms.

Comparing Accuracies             |  Comparing Losses
:-------------------------:|:-------------------------:
![](https://github.com/anubhav2901/Adaptive-Gradient-Descent/blob/main/figures/accuracies.png)  |  ![](https://github.com/anubhav2901/Adaptive-Gradient-Descent/blob/main/figures/Losses.png)


## Requirements.txt

> - matplotlib==3.5.2
> - numpy==1.22.3
> - scikit_learn==1.1.1
> - scipy==1.8.0

## Demonstration
The implementation can tested by running the main.py as in the command shown:

> python main.py

## References

[1] Yura Malitsky, Konstantin Mishchenko Proceedings of the 37th International Conference on Machine Learning, PMLR 119:6702-6712, 2020.
[2] https://arxiv.org/pdf/1910.09529v2.pdf
