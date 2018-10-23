# Machine Learning Resources - Becoming a Machine Learning Expert
From Basic to Advanced

* [1. Teaser/ Foundation/ Prerequisites Neuronal Networks](#1-teaser-foundation-prerequisites-neural-networks)
* [2. Prerequisites Math](#2-prerequisites-math)
	* [2.1 Linear Regression](#21-linear-regression)
	* [2.2 Logistic Regression](#22-logistic-regression)
	* [2.3 K Nearest Neighbors](#23-k-nearest-neighbors)
	* [2.4 K-Means](#24-k-means)
	* [2.5 Decision Trees](#25-decision-trees)
* [3. Prerequisites Programming](#3-prerequisites-programming)
	* [3.1 Python](#31-python)
	* [3.2 Mathlab](#32-mathlab)
	* [3.3 Jupyter Notebooks](#33-jupyter-notebooks)
* [NLP](#nlp)
* [Deep Reinforcement Learning](#deep-reinforcement-learning)
* [ML Project Advice](#ml-project-advice)


## 1. Teaser/ Foundation/ Prerequisites Neuronal Networks: 

* [Machine Learning Introduction](https://www.youtube.com/watch?v=seG9J49bBYI): Loved this video because it starts with great definitions and introduces you to important terminology. Feel free to stop at 6:47.
* [What is Machine Learning?](https://www.youtube.com/watch?v=WXHM_i-fgGo): Explains the 3 different subareas of machine learning: supervised learning, unsupervised learning, reinforcement learning
*  [A Friendly Intro to Machine Learning](https://www.youtube.com/watch?v=IpGxLWOIZy4): Great video with some cool illustrations, but honestly I think just watching until 5:54 is sufficient. 
* [Basic Machine Learning Algorithms Overview](https://www.youtube.com/watch?v=ggIk08PNcBo): Don't worry about knowing exactly what every one of these terms mean. Just get a sense for the different algorithms and the tasks they are used for. We'll go into way more detail later on. 
* [Machine Learning from Zero to Hero](https://medium.freecodecamp.org/machine-learning-how-to-go-from-zero-to-hero-40e26f8aa6da): Get motivated to learn Machine Learning.  Sometimes a simple sitting back and watching the right videos can ignite and fan the fire to get active in ML.  If you know software, this is a great starting post. Don't worry about knowing every single detail in each of the videos, but rather think about the high level goal of ML. 
* [The Master Algorithm - Petro Domingos](https://www.amazon.de/Master-Algorithm-Ultimate-Learning-Machine/dp/0141979240/ref=sr_1_1?ie=UTF8&qid=1538107975&sr=8-1&keywords=master+algorithm)

## 2. Prerequisites Math:
*Recommended Books:

* [Matrix Calculus](http://parrt.cs.usfca.edu/doc/matrix-calculus/index.html)
* [Mathematics for Machine Learning](http://gwthomas.github.io/docs/math4ml.pdf)

### 2.1 **Linear Regression**: 
* [Coursera Course - Linear Regression with One Variable](https://www.youtube.com/watch?v=kHwlB_j7Hkc)
* [Simple Linear Regression](https://www.youtube.com/watch?v=ZkjP5RJLQF4)
* [Linear Regression for Machine Learning](https://machinelearningmastery.com/linear-regression-for-machine-learning/)

### 2.2 **Logistic Regression**: 
* [Coursera Course - Logistic Regression and Classification](https://www.youtube.com/watch?v=-la3q9d7AKQ)
* [Logistic Regression - An Introduction](https://www.youtube.com/watch?v=zAULhNrnuL4)
* [Stanford Logistic Regression Overview](http://ufldl.stanford.edu/tutorial/supervised/LogisticRegression/)
* [Siraj Raval Logistic Regression Tutorial](https://www.youtube.com/watch?v=D8alok2P468)

### 2.3 **K Nearest Neighbors**: 
* [K Nearest Neighbors](https://www.youtube.com/watch?v=k_7gMp5wh5A)
* [GeeksForGeeks Explanation](https://www.geeksforgeeks.org/k-nearest-neighbours/)
* [Udacity Explanation of KNNs](https://www.youtube.com/watch?v=mpU84OJ5vdQ)

### 2.4 **K-Means**: 
* [Coursera Course - K Means](https://www.youtube.com/watch?v=hDmNF9JG3lo)
* [Stanford K Means Overview](http://stanford.edu/~cpiech/cs221/handouts/kmeans.html)
* [Siraj Raval K Means Tutorial](https://www.youtube.com/watch?v=9991JlKnFmk)

### 2.5 **Decision Trees**:
* [Nando de Freitas Lecture](https://www.youtube.com/watch?v=-dCtJjlEEgM)
* [Decision Trees in ML](https://towardsdatascience.com/decision-trees-in-machine-learning-641b9c4e8052)

**Optional**: Random Forest, SVMs, Naive Bayes, Gradient Boosted Methods, PCA

## 3. Prerequisites Programming:
### 3.1 **Python**:

### 3.2 **Mathlab**:

#### 3.3 **Jupyter Notebooks**:



## 4. Deep Learning Frameworks:

### 4.1 **[Keras](https://keras.io/)**:

* [Tensorflow](https://www.tensorflow.org/) - This is my go-to deep library nowdays. Honestly I think it has the steepest learning curve because it takes quite a while to get comfortable with the ideas of Tensorflow variables, placeholders, and building/executing graphs. One of the big plus sides to Tensorflow is the number of Github and Stackoverflow help you can get. You can find the answer to almost any error you get in Tensorflow because someone has likely run into it before. I think that's hugely helpful. 
* [Torch](http://torch.ch/) - 2015 was definitely the year of Torch, but unless you really want to learn Lua, PyTorch is probably the way to go now. However, there’s a lot of good documentation and tutorials associated with Torch, so that’s a good upside. 
* [PyTorch](http://pytorch.org/) - My other friend and I have this joke where we say that if you’re running into a bug in PyTorch, you could probably read the entirety of PyTorch’s documentation in less than 2 hours and you still wouldn’t find your answer LOL. But honestly, so many AI researchers have been raving about it, so it’s definitely worth giving it a shot even though it’s still pretty young. I think Tensorflow and PyTorch will be the 2 frameworks that will start to take over the DL framework space.
* [Caffe](http://caffe.berkeleyvision.org/) and [Caffe2](https://caffe2.ai/) - Never played around with Caffe, but this was one of the first deep learning libraries out there. Caffe2 is notable because it's the production framework that Facebook uses to serve its models. [According to Soumith Chintala](https://www.oreilly.com/ideas/why-ai-and-machine-learning-researchers-are-beginning-to-embrace-pytorch), researchers at Facebook will try out new models and research ideas using PyTorch and will deploy using Caffe2. 

## 5. Basics Machine Learning (Deep learning):
Supervised, Unsupervised, Reinforcement learning

### 5.1 **Artificial Neural Networks**: 
If someone wants to get started with deep learning, I think that the best approach is to first get familiar with machine learning (which you all will have done by this point) and then start with neural networks. Following the same high level understanding -> model specifics -> code -> practical example approach would be great here as well. 
* [3Blue1Brown Neural Network Playlist](https://www.youtube.com/watch?v=aircAruvnKk&list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi): Can't overstate how well made these videos are. They go in depth on backpropagation and gradient descent, two critical concepts for understanding neural networks. 
* [How Deep Neural Networks Work](https://www.youtube.com/watch?v=ILsA4nyG7I0): Another great tutorial by Brandon Rohrer.
* [A Friendly Introduction to Deep Learning and Neural Networks](https://www.youtube.com/watch?v=BR9h47Jtqyw): Another visually appearing presentation of neural nets.
* [Neural Network Playground](http://playground.tensorflow.org/#activation=tanh&batchSize=10&dataset=circle&regDataset=reg-plane&learningRate=0.03&regularizationRate=0&noise=0&networkShape=4,2&seed=0.09661&showTestData=false&discretize=false&percTrainData=50&x=true&y=true&xTimesY=false&xSquared=false&ySquared=false&cosX=false&sinX=false&cosY=false&sinY=false&collectStats=false&problem=classification&initZero=false&hideText=false): Great web app, created by Google, that allows you to tinker with a neural network in the browser. Awesome for gaining some practical understanding. 
	* [Michael Nielsen Book on NNs](http://neuralnetworksanddeeplearning.com/chap1.html): Very in depth and comprehensive.

## 5.2 **Convolutional Neural Networks**:  
A convolutional neural network is a special type of neural network that has been successfully used for image processing tasks.
* [A Beginner's Guide to Understanding CNNs](https://adeshpande3.github.io/adeshpande3.github.io/A-Beginner's-Guide-To-Understanding-Convolutional-Neural-Networks/): Shameless plug LOL
* [CS 231N Homepage](http://cs231n.github.io/convolutional-networks/): Stanford CS231N is a grad course focused on CNNs that was originally taught by Fei Fei Li, Andrej Karpathy, and others.
* [CS 231N Video Lectures](https://www.youtube.com/playlist?list=PLkt2uSq6rBVctENoVBg1TpCC7OQi31AlC): All the lecture videos from 2016. There will likely be a playlist for 2017 somewhere on YouTube as well. 
* [Brandon Rohrer YouTube Tutorial](https://www.youtube.com/watch?v=FmpDIaiMIeA): Great visuals on this tutorial video. 
* [Andrew Ng's CNN Course](https://www.youtube.com/playlist?list=PLBAGcD3siRDjBU8sKRk0zX9pMz9qeVxud): Videos from Andrew Ng's deep learning course.
* [Stanford CS 231N](https://www.youtube.com/watch?v=g-PvXUjD6qg&list=PLlJy-eBtNFt6EuMxFYRiNRS07MCWN5UIA) - CNNs
* [Feature Visualization](https://distill.pub/2017/feature-visualization/) - Feature VisualizationHow neural networks build up their understanding of images

## 5.3 **Recurrent Neural Networks**:
A recurrent neural network is a special type of neural network that has been successfully used for natural language processing tasks.
* [Deep Learning Research Paper Review: NLP](https://adeshpande3.github.io/adeshpande3.github.io/Deep-Learning-Research-Review-Week-3-Natural-Language-Processing): Too many shameless plugs or nah? LOL
* [CS 224D Video Lectures](https://www.youtube.com/playlist?list=PLCJlDcMjVoEdtem5GaohTC1o9HTTFtK7_): Stanford CS 224D is a grad course focused on RNNs and applying deep learning to NLP. 
* [RNNs and LSTMs](https://www.youtube.com/watch?v=WCUNPb-5EYI): We all love Brandon honestly. 
* [Recurrent Neural Networks - Intel Nervana](https://www.youtube.com/watch?v=Ukgii7Yd_cU): Very comprehensive.
* [Understanding LSTM Networks](http://colah.github.io/posts/2015-08-Understanding-LSTMs/): Chris Olah's posts are readable, yet in-depth.
* [Introduction to RNNs](http://www.wildml.com/2015/09/recurrent-neural-networks-tutorial-part-1-introduction-to-rnns/): Denny Britz is another great author who has a wide ranging blog.

	d) **Reinforcement Learning**: While the 3 prior ML methods are necessarily important for understanding RL, a lot of recent progress in this field has combined elements from the deep learning camp as well as from the traditional reinforcement learning field. 
	* [David Silver's Reinforcement Learning Course](https://www.youtube.com/watch?v=2pWv7GOvuf0&list=PL5X3mDkKaJrL42i_jhE4N-p6E2Ol62Ofa): Advanced stuff covered here, but David is a fantastic lecturer and I loved the comprehensive content. 
	* [Simple Reinforcement Learning with Tensorflow](https://medium.com/emergent-future/simple-reinforcement-learning-with-tensorflow-part-0-q-learning-with-tables-and-neural-networks-d195264329d0): Arthur Juliani has a blog post series that covers RL concepts with lots of practical examples.
	* [David Silver's Reinforcement Learning Course](https://www.youtube.com/watch?v=2pWv7GOvuf0&list=PL5X3mDkKaJrL42i_jhE4N-p6E2Ol62Ofa)
	* [Deep Reinforcement Learning Doesn't Work Yet](https://www.alexirpan.com/2018/02/14/rl-hard.html)
	* [Deep RL Arxiv Review Paper](https://arxiv.org/pdf/1701.07274v2.pdf)
	* [Pong From Pixels](http://karpathy.github.io/2016/05/31/rl/)
	* [Lessons Learned Reproducing a Deep RL Paper](http://amid.fish/reproducing-deep-rl)
	
	e) Kaggle
	* [blog](http://blog.kaggle.com/)
	
	f) Pretrained Models

## 6. Advanced Machine Learning: 
### 6.1 Python Libraries for ML:
	* [Numpy](http://www.numpy.org/)
	* [Scikit-Learn](http://scikit-learn.org/stable/)
	* [Matplotlib](https://matplotlib.org/)
	* [Pandas](http://pandas.pydata.org/)
### 6.2 Advanced Neuronal Background Knowledge
### 6.3 Advanced Mathematical Background Knowledge:

## 7. Machine Learning Research (Go deeper):

## Best Courses


* [Stanford CS 224D](https://www.youtube.com/watch?v=sU_Yu_USrNc&list=PLTuSSFCVeNVCXL0Tak5rJ83O-Bg_ajO5B) - Deep Learning for NLP
* [Hugo Larochelle's Neural Networks Course](https://www.youtube.com/watch?v=SGZ6BttHMPw&list=PL6Xpj9I5qXYEcOhn7TqghAJ6NAPrNmUBH)

* [Andrew Ng Machine Learning Course](https://www.coursera.org/learn/machine-learning)
* [Stanford CS 229](https://www.youtube.com/playlist?list=PLA89DCFA6ADACE599) - Pretty much the same as the Coursera course
* [UC Berkeley Kaggle Decal](https://kaggledecal.github.io/)
* [Short MIT Intro to DL Course](https://www.youtube.com/playlist?list=PLkkuNyzb8LmxFutYuPA7B4oiMn6cjD6Rs)
* [Udacity Deep Learning](https://www.udacity.com/course/deep-learning--ud730)
* [Deep Learning School Montreal 2016](http://videolectures.net/deeplearning2016_montreal/) and [2017](http://videolectures.net/deeplearning2017_montreal/)
* [Intro to Neural Nets and ML (Univ of Toronto)](http://www.cs.toronto.edu/~rgrosse/courses/csc321_2017/)
* [Deep RL Bootcamp](https://sites.google.com/view/deep-rl-bootcamp/lectures)
* [CMU Neural Networks for NLP](http://www.phontron.com/class/nn4nlp2017/schedule.html)
* [Bay Area Deep Learning School Day 1 2016](https://www.youtube.com/watch?v=eyovmAtoUx0) and [Day 2](https://www.youtube.com/watch?v=9dXiAecyJrY)
* [Introduction to Deep Learning MIT Course](https://www.youtube.com/playlist?list=PLkkuNyzb8LmxFutYuPA7B4oiMn6cjD6Rs)
* [Caltech CS 156 - Machine Learning](https://www.youtube.com/playlist?list=PLD63A284B7615313A)
* [Berkeley EE 227C - Convex Optimization](https://ee227c.github.io/)

## Most Important Deep Learning Papers

* [AlexNet](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf)
* [GoogLeNet](https://arxiv.org/pdf/1409.4842v1.pdf)
* [VGGNet](https://arxiv.org/pdf/1409.1556v6.pdf)
* [ZFNet](https://arxiv.org/pdf/1311.2901v3.pdf)
* [ResNet](https://arxiv.org/pdf/1512.03385.pdf)
* [R-CNN](https://arxiv.org/pdf/1311.2524v5.pdf)
* [Fast R-CNN](https://arxiv.org/pdf/1504.08083.pdf)
* [Adversarial Images](https://arxiv.org/pdf/1412.1897.pdf)
* [Generative Adversarial Networks](https://arxiv.org/pdf/1406.2661v1.pdf)
* [Spatial Transformer Networks](https://arxiv.org/pdf/1506.02025.pdf)
* [DCGAN](https://arxiv.org/pdf/1511.06434v2.pdf)
* [Synthetic Gradients](https://arxiv.org/pdf/1608.05343v1.pdf)
* [Memory Networks](https://arxiv.org/pdf/1410.3916v11.pdf)
* [Mixture of Experts](https://arxiv.org/pdf/1701.06538.pdf)
* [Neural Turing Machines](https://arxiv.org/pdf/1410.5401.pdf)
* [Alpha Go](https://storage.googleapis.com/deepmind-media/alphago/AlphaGoNaturePaper.pdf)
* [Atari DQN](https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf)
* [Word2Vec](https://arxiv.org/pdf/1310.4546.pdf)
* [GloVe](https://nlp.stanford.edu/pubs/glove.pdf)
* [A3C](https://arxiv.org/pdf/1602.01783v2.pdf)
* [Gradient Descent by Gradient Descent](https://arxiv.org/pdf/1606.04474v1.pdf)
* [Rethinking Generalization](https://arxiv.org/pdf/1611.03530v1.pdf)
* [Densely Connected CNNs](https://arxiv.org/pdf/1608.06993v1.pdf)
* [EBGAN](https://arxiv.org/pdf/1609.03126v1.pdf)
* [Wasserstein GAN](https://arxiv.org/pdf/1701.07875.pdf)
* [Style Transfer](https://arxiv.org/pdf/1603.08155v1.pdf)
* [Pixel RNN](https://arxiv.org/pdf/1601.06759v2.pdf)
* [Dynamic Coattention Networks](https://arxiv.org/pdf/1611.01604v2.pdf)
* [Convolutional Seq2Seq Learning](https://arxiv.org/pdf/1705.03122.pdf)
* [Seq2Seq](https://arxiv.org/pdf/1409.3215.pdf)
* [Dropout](https://www.cs.toronto.edu/~hinton/absps/JMLRdropout.pdf)
* [Batch Norm](https://arxiv.org/pdf/1502.03167.pdf)
* [Large Batch Training](https://arxiv.org/pdf/1609.04836.pdf)
* [Transfer Learning](http://papers.nips.cc/paper/5347-how-transferable-are-features-in-deep-neural-networks.pdf)
* [Adam](https://arxiv.org/pdf/1412.6980.pdf)
* [Speech Recognition](https://arxiv.org/pdf/1303.5778.pdf)
* [Relational Networks](https://arxiv.org/pdf/1706.01427.pdf)
* [Influence Functions](https://arxiv.org/pdf/1703.04730.pdf)
* [ReLu](https://arxiv.org/pdf/1611.01491.pdf)
* [Xavier Initialization](http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf)
* [Saddle Points and Non-convexity of Neural Networks](https://arxiv.org/pdf/1406.2572.pdf)
* [Overcoming Catastrophic Forgetting in NNs](https://arxiv.org/pdf/1612.00796.pdf)
* [Quasi-Recurrent Neural Networks](https://arxiv.org/pdf/1611.01576.pdf)
* [Escaping Saddle Points Efficiently](https://arxiv.org/pdf/1703.00887.pdf)
* [Progressive Growing of GANs](http://research.nvidia.com/sites/default/files/pubs/2017-10_Progressive-Growing-of//karras2017gan-paper.pdf)
* [Attention is All You Need](https://arxiv.org/pdf/1706.03762.pdf)
* [Dynamic Routing Between Capsules](https://arxiv.org/pdf/1710.09829.pdf)
* [Unsupervised Machine Translation with Monolingual Corpora](https://arxiv.org/pdf/1711.00043.pdf)
* [Population Based Training of NN's](https://arxiv.org/pdf/1711.09846.pdf)
* [Learned Index Structures](https://arxiv.org/pdf/1712.01208v1.pdf)
* [Visualizing Loss Landscapes](https://arxiv.org/pdf/1712.09913v1.pdf)
* [DenseNet](https://arxiv.org/pdf/1608.06993.pdf)
* [SqueezeNet](https://arxiv.org/pdf/1602.07360.pdf)
* [WaveNet](https://arxiv.org/pdf/1609.03499.pdf)
* [Hidden Technical Debt in ML Systems](https://papers.nips.cc/paper/5656-hidden-technical-debt-in-machine-learning-systems.pdf)
* [MobileNets](https://arxiv.org/pdf/1704.04861.pdf)
* [Learning from Imbalanced Data](http://www.ele.uri.edu/faculty/he/PDFfiles/ImbalancedLearning.pdf)

## History of Deep Learning Papers


## Stay tuned
1) Blogs
-add blocks from universitys
* [Andrej Karpathy](http://karpathy.github.io/)
* [Google Research](https://research.googleblog.com/)
* [Neil Lawrence](http://inverseprobability.com/blog)
* [Qure.ai](http://blog.qure.ai/)
* [Brandon Amos](http://bamos.github.io/blog/)
* [Denny Britz](http://www.wildml.com/)
* [Moritz Hardt](http://blog.mrtz.org/)
* [Deepmind](https://deepmind.com/blog/)
* [Machine Learning Mastery](http://machinelearningmastery.com/blog/)
* [Smerity](http://smerity.com/articles/articles.html)
* [The Neural Perspective](https://theneuralperspective.com/)
* [Pete Warden](https://petewarden.com/page/2/)
* [Kevin Zakka](https://kevinzakka.github.io/)
* [Thomas Dinsmore](https://thomaswdinsmore.com/)
* [Rohan Varma](http://rohanvarma.me/)
* [Anish Athalye](https://www.anishathalye.com/)
* [Arthur Juliani](https://medium.com/@awjuliani)
* [CleverHans](http://www.cleverhans.io/)
* [Off the Convex Path](http://www.offconvex.org/about/)
* [Sebastian Ruder](http://ruder.io/#open)
* [Berkeley AI Research](http://bair.berkeley.edu/blog/)
* [Facebook AI Research](https://research.fb.com/blog/)
* [Salesforce Research](https://www.salesforce.com/products/einstein/ai-research/)
* [Apple Machine Learning Journal](https://machinelearning.apple.com/)
* [OpenAI](https://blog.openai.com/)
* [Lab41](https://gab41.lab41.org/tagged/machine-learning)
* [Depth First Learning](http://www.depthfirstlearning.com/)
* [The Gradient](https://thegradient.pub/)
* [Distill](https://distill.pub/)

2) Talks
Potcasts
Blogs





 



