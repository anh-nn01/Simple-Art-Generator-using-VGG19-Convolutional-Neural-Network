# @author Anh Nhu
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import imageio
import png

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)


image_shape = imageio.imread("Beauty.jpg").shape
image = tf.keras.preprocessing.image.load_img("Beauty.jpg", target_size = image_shape)

plt.imshow(image)
plt.title("Sample Image")
plt.show()

"""
Preprocess the input
"""
img = tf.keras.preprocessing.image.img_to_array(image)
img = np.expand_dims(img, axis = 0)
img = tf.keras.applications.vgg19.preprocess_input(img)

"""
Return the activations in convolution layers in VGG19
"""
def model_activations(img, input):

    model = tf.keras.applications.VGG19(include_top = False, weights = "imagenet")
    model.trainable = False # we will not change the paramenters of the VGG19
    model.summary()

    res = {}

    """
    Define layers of VGG19 from Tensorflow
    """
    input = model.get_layer(name = input)(img)

    block1_conv1_features = model.get_layer(name = "block1_conv1")(input)
    block1_conv2_features = model.get_layer(name = "block1_conv2")(block1_conv1_features)
    block1_pool_features  = model.get_layer(name = "block1_pool")(block1_conv2_features)

    block2_conv1_features = model.get_layer(name = "block2_conv1")(block1_pool_features)
    block2_conv2_features = model.get_layer(name = "block2_conv2")(block2_conv1_features)
    block2_pool_features  = model.get_layer(name = "block2_pool")(block2_conv2_features)

    block3_conv1_features = model.get_layer(name = "block3_conv1")(block2_pool_features)
    block3_conv2_features = model.get_layer(name = "block3_conv2")(block3_conv1_features)
    block3_conv3_features = model.get_layer(name = "block3_conv3")(block3_conv2_features)
    block3_conv4_features = model.get_layer(name = "block3_conv4")(block3_conv3_features)
    block3_pool_features  = model.get_layer(name = "block3_pool")(block3_conv4_features)

    block4_conv1_features = model.get_layer(name = "block4_conv1")(block3_pool_features)
    block4_conv2_features = model.get_layer(name = "block4_conv2")(block4_conv1_features)
    block4_conv3_features = model.get_layer(name = "block4_conv3")(block4_conv2_features)
    block4_conv4_features = model.get_layer(name = "block4_conv4")(block4_conv3_features)
    block4_pool_features = model.get_layer(name  = "block4_pool")(block4_conv4_features)

    block5_conv1_features = model.get_layer(name = "block5_conv1")(block4_conv4_features)
    block5_conv2_features = model.get_layer(name = "block5_conv2")(block5_conv1_features)
    block5_conv3_features = model.get_layer(name = "block5_conv3")(block5_conv2_features)
    block5_conv4_features = model.get_layer(name = "block5_conv4")(block5_conv3_features)
    block5_pool_features  = model.get_layer(name = "block5_pool")(block5_conv4_features)

    res["b1_conv1_activation"] = block1_conv1_features
    res["b1_conv2_activation"] = block1_conv2_features
    res["b1_pool_activation"]  = block1_pool_features

    res["b2_conv1_activation"] = block2_conv1_features
    res["b2_conv2_activation"] = block2_conv2_features
    res["b2_pool_activation"]  = block2_pool_features

    res["b3_conv1_activation"] = block3_conv1_features
    res["b3_conv2_activation"] = block3_conv2_features
    res["b3_conv3_activation"] = block3_conv3_features
    res["b3_conv4_activation"] = block3_conv4_features
    res["b3_pool_activation"]  = block3_pool_features

    res["b4_conv1_activation"] = block4_conv1_features
    res["b4_conv2_activation"] = block4_conv2_features
    res["b4_conv3_activation"] = block4_conv3_features
    res["b4_conv4_activation"] = block4_conv4_features
    res["b4_pool_activation"]  = block4_pool_features

    res["b5_conv1_activation"] = block5_conv1_features
    res["b5_conv2_activation"] = block5_conv2_features
    res["b5_conv3_activation"] = block5_conv3_features
    res["b5_conv4_activation"] = block5_conv4_features
    res["b5_pool_activation"]  = block5_pool_features

    return res

"""
Define the Loss / Optimization objective for drawing an image
"""
def Loss(activation_product, activation_content):
    n_H = activation_product.shape[1] # vertical dimension of a channel in current activation layer
    n_W = activation_product.shape[2] # horizontal dimension of a channel in current activation layer
    n_C = activation_product.shape[3] # number of channels in current activation layer
    # @author Anh Nhu
    loss = tf.reduce_sum(tf.pow(activation_product - activation_content, 2))
    loss = (1 / (4 * n_H * n_W * n_C)) * loss

    return loss

"""
Draw the picture
"""
def Draw(prod, epoch, layer):
  with tf.GradientTape() as tape:
      # @author Anh Nhu
      # Layer activations of the product through VGG19
      product_activation = model_activations(prod, input = "input_" + str(epoch+2))

      loss = Loss(product_activation[layer], activation[layer])

      grad = tape.gradient(loss, prod)
      # Update the pixels
      opt.apply_gradients([(grad, prod)])

      # Clip the pixel values that fall outside the range of [0,1]
      prod.assign(tf.clip_by_value(prod, clip_value_min=0.0, clip_value_max=1.0))

      #show resulting image after each epoch
      plt.imshow(prod[0,:,:,:])
      plt.title("Drawing... Epoch " + str(epoch))
      plt.show()
      print(loss)

"""
Create an empty template to draw
"""
def create_template():
    img_temp = np.ones((image_shape[0], image_shape[1], image_shape[2])) * 255
    product = Image.fromarray(img_temp.astype('uint8')).convert('RGB')
    product = tf.keras.preprocessing.image.img_to_array(product)
    product = np.expand_dims(product, axis = 0)
    product = tf.keras.applications.vgg19.preprocess_input(product)
    product = tf.Variable(product)

    return product


activation  = model_activations(img, input = "input_1")
opt = tf.optimizers.Adam(learning_rate = 0.7)
product = create_template()


plt.imshow(product[0,:,:,:])
plt.title("Blank paper")
plt.show()

for i in range(0, 6):
    Draw(product, i, layer = "b5_conv2_activation")
for i in range(6, 12):
    Draw(product, i, layer = "b5_conv1_activation")

opt = tf.optimizers.Adam(learning_rate = 0.6)
for i in range(12, 22):
    Draw(product, i, layer = "b2_conv2_activation")

opt = tf.optimizers.Adam(learning_rate = 0.1)
for i in range(22, 23):
    Draw(product, i, layer = "b1_conv2_activation")

mean = [103.939/255., 116.779/255., 123.68/255.]

prod = product.numpy()

# Convert the image back to RGB standard
# Zero-center by mean pixel
prod[..., 0] += mean[0]
prod[..., 1] += mean[1]
prod[..., 2] += mean[2]
prod = prod[..., ::-1]
# @author Anh Nhu
# Display the final product
plt.imshow(prod[0,:,:,:])
plt.title("Product")
plt.show()
