# Simple-Art-Generator-using-VGG19-Convolutional-Neural-Network
In this project, I try a simple application of Deep Neural Network: drawing art using pretrained model. The model is the VGG19 Convnet proposed Karen Simonyan and Andrew Zisserman. The objective of the project is: given a random input, which is an image of any kind (car, girl, nature, etc), the convolutional neural network will be utilized to analyze the image and try to draw it from a blank paper. In this project, I have slowed down the drawing process to show how the neural network draw the picture step-by-step. In other words, how the drawing starts looking like the original picture.

The intuition is that a convolutional neural network has been train to recognize objects pretty well, especially the one with large structure like the VGG19. Once we have the brain, something that have learned to "conceive" objects, we can use that "brain" to do many tasks, such as drawing. The brain here is the pretrained VGG19. You can take a look at my project "A-Close-Examination-of-Activations-in-Deep-Convolutional-Neural-Network--Visualiazing-Deep-ConvNet" to have a better intuition. If you want to understand how to teach the brain to conceive things, or to train a neural network from scratch, please look at my first project "Neural-Network-from-Scratch--Hand-written-Digits-classifier" for some math explanations.

Explanation of the main idea: normally, we try to optimize parameters in neural network to have the lowest cost. In this project, parameters are the pixels in the blank paper, and the cost measure how different the activations of the current product with those of the original picture. In convolutional neural network, the shallower layers tried to recognize the simple features, and deeper layers look for abstract features. In my implementation, the algorithm tries to optimize the drawing by looking at deep layers first and then shallow layers. As a result, you can see when it tried to draw a girl's face, it first tried to capture abstract features such as eyes, noses, comlex facial shapes, etc. However, at abstract level, it only showed the general shapes of such objects. When the algorithm moved back to shallower layers for optimizations, the picture already had complex objects (but without complex details). Since the shallow layers can recognize simple characteristics such as vertical edges, horizontal edges, curves, etc, it added a great amount of details to the picture, hence making it looks more realistic. This is exactly the same how we draw a picture: we draw general, abstract things first and then add details and color the picture. You can download the code to play around and test what happens if you change activation layers or simply just optimize on shallow layers first and then deep layers. Rememeber to have your images downloaded in the pythonscripts file, and I recommend unless you use a very powerful GPU, you should test the algorith with less than 30 epochs and with pictures with low resolution.

**Link to the video showing how the NN learn to draw the picture: https://photos.app.goo.gl/3nbhp69F81kCLX1j8**

* **Note that the point is not to imitate the picture, but to draw the picture from a blank paper so that it looks like a DRAWING of the picture, not an imitation! This explains why the final result does not look 100% like the original one, but more like a drawing!**

Here are some images showing how the NN draw the picture:

**This picture is what the algorithm have to draw**

<img src = "Images/Sample.png">

**This is how the algorithm did it:**

<img src = "Images/Blank paper.png">
<img src = "Images/Epoch 1.png">
<img src = "Images/Epoch 2.png">
<img src = "Images/Epoch 3.png">
<img src = "Images/Epoch 4.png">
<img src = "Images/Epoch 5.png">
<img src = "Images/Epoch 6.png">
<img src = "Images/Epoch 7.png">
<img src = "Images/Epoch 8.png">
<img src = "Images/Epoch 9.png">
<img src = "Images/Epoch 10.png">
<img src = "Images/Epoch 11.png">

Before training the neural network, the inputs were preprocessed. As a result, the output in this case (the drawing) is also shown in the preprocessed form. Therefore, we need to reverese such preprocessing to obtain the final drawing:

<img src = "Images/Final Drawing.png">


**Further application of this algorithm:**

This algorithm can be employed in the Neural Style Transfer as the content-optimizer in that project. Beside the content-optimizer, which focuses on the main objects in the picture (people, animals, cars, etc), we also need a style-optimizer algorithm, which analyzes the style of famous pictures (Picasso's, Da Vinci's, abstract art, etc) and tries to make art base on such style.
