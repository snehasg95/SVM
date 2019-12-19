# SVM
Support Vector Machines


Kernel Trick
*  The idea is mapping the non-linear separable data-set into a higher dimensional space where we can find a hyperplane that can separate the samples.


Kernel Types:
* Linear
* Radial Basis Function (RBF)

Which Kernel to use?

* So, the rule of thumb is: use linear SVMs (or logistic regression) for linear problems, and nonlinear kernels such as the Radial Basis Function kernel for non-linear problems.

https://www.kdnuggets.com/2016/06/select-support-vector-machine-kernels.html

When the RBF is chosen the following parameters can be tuned

Parameter Tuning

* Gamma
* C



Intuitively, the gamma parameter defines how far the influence of a single training example reaches, with low values meaning ‘far’ and high values meaning ‘close’. The gamma parameters can be seen as the inverse of the radius of influence of samples selected by the model as support vectors.

Keep Gamma higer to have a better decision area ** But definetely not too high as it results in overfitting. See this link for graphs as how this is affected.

https://chrisalbon.com/machine_learning/support_vector_machines/svc_parameters_using_rbf_kernel/


Regularization parameter in python's Scikit-learn C parameter used to maintain regularization. Here C is the penalty parameter, which represents misclassification or error term. The misclassification or error term tells the SVM optimization how much error is bearable. This is how you can control the trade-off between decision boundary and misclassification term. 
For larger values of C, a smaller margin will be accepted if the decision function is better at classifying all training points correctly. A lower C will encourage a larger margin, therefore a simpler decision function, at the cost of training accuracy. In other words``C`` behaves as a regularization parameter in the SVM.


Keep a very high C value to ,make sure there is no variance in the decision boundry margin.


Advantages

SVM Classifiers offer good accuracy and perform faster prediction compared to Naïve Bayes algorithm. They also use less memory because they use a subset of training points in the decision phase. SVM works well with a clear margin of separation and with high dimensional space.
Disadvantages

SVM is not suitable for large datasets because of its high training time and it also takes more time in training compared to Naïve Bayes. It works poorly with overlapping classes and is also sensitive to the type of kernel used.


https://www.datacamp.com/community/tutorials/svm-classification-scikit-learn-python
