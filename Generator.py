import tensorflow as tf
from tensorflow.python import pywrap_tensorflow
import numpy as np
from tensorflow import keras
WEIGHT_INIT_STDDEV = 0.05


class Generator(object):

	def __init__(self, sco):
		self.encoder = Encoder(sco)
		self.decoder = Decoder(sco)

	def transform(self, vis, ir):
		img = tf.concat([vis, ir], 3)
		code = self.encoder.encode(img)
		self.target_features = code
		generated_img = self.decoder.decode(self.target_features)
		return generated_img


class Encoder(object):
	def __init__(self, scope_name):
		self.scope = scope_name
		self.weight_vars = []
		with tf.compat.v1.variable_scope(self.scope):
			with tf.compat.v1.variable_scope('encoder'):
				self.weight_vars.append(self._create_variables(2, 48, 3, scope = 'conv1_1'))
				self.weight_vars.append(self._create_variables(48, 48, 3, scope = 'dense_block_conv1'))
				self.weight_vars.append(self._create_variables(96, 48, 3, scope = 'dense_block_conv2'))
				self.weight_vars.append(self._create_variables(144, 48, 3, scope = 'dense_block_conv3'))
				self.weight_vars.append(self._create_variables(192, 48, 3, scope = 'dense_block_conv4'))
				self.condense_block = CondenseNetV2(240, 48, 3, scope='condense_block')
	def _create_variables(self, input_filters, output_filters, kernel_size, scope):
		shape = [kernel_size, kernel_size, input_filters, output_filters]
		with tf.compat.v1.variable_scope(scope):
			kernel = tf.Variable(tf.compat.v1.truncated_normal(shape, stddev = WEIGHT_INIT_STDDEV),
			                     name = 'kernel')
			bias = tf.Variable(tf.zeros([output_filters]), name = 'bias')
		return (kernel, bias)

	def encode(self, image):
		dense_indices = [1, 2, 3, 4, 5]
		out = image
		for i in range(len(self.weight_vars)):
			kernel, bias = self.weight_vars[i]
			if i in dense_indices:
				out = conv2d(out, kernel, bias, dense=True, use_relu=True,
							 Scope=self.scope + '/encoder/b' + str(i))
			else:
				out = conv2d(out, kernel, bias, dense=False, use_relu=True,
							 Scope=self.scope + '/encoder/b' + str(i))

		#out = fpn(out, 240)
		out = self.condense_block.forward(out)
		out = CBAM_attention(out)

		return out


class Decoder(object):
	def __init__(self, scope_name):
		self.weight_vars = []
		self.scope = scope_name
		with tf.name_scope(scope_name):
			with tf.compat.v1.variable_scope('decoder'):
				self.weight_vars.append(self._create_variables(384, 384, 3, scope = 'conv2_1'))
				self.weight_vars.append(self._create_variables(384, 128, 3, scope = 'conv2_2'))
				self.weight_vars.append(self._create_variables(128, 64, 3, scope = 'conv2_3'))
				self.weight_vars.append(self._create_variables(64, 32, 3, scope = 'conv2_4'))
				self.weight_vars.append(self._create_variables(32, 1, 3, scope = 'conv2_5'))


	def _create_variables(self, input_filters, output_filters, kernel_size, scope):
		with tf.compat.v1.variable_scope(scope):
			shape = [kernel_size, kernel_size, input_filters, output_filters]
			kernel = tf.Variable(tf.compat.v1.truncated_normal(shape, stddev = WEIGHT_INIT_STDDEV), name = 'kernel')
			bias = tf.Variable(tf.zeros([output_filters]), name = 'bias')
		return (kernel, bias)

	def decode(self, image):
		final_layer_idx = len(self.weight_vars) - 1

		out = image

		for i in range(len(self.weight_vars)):
			kernel, bias = self.weight_vars[i]
			if i == 0:
				out = conv2d(out, kernel, bias, dense = False, use_relu = True,
				             Scope = self.scope + '/decoder/b' + str(i), BN = False)
			if i == final_layer_idx:
				out = conv2d(out, kernel, bias, dense = False, use_relu = False,
				             Scope = self.scope + '/decoder/b' + str(i), BN = False)
				out = tf.nn.tanh(out) / 2 + 0.5
			else:
				out = conv2d(out, kernel, bias, dense = False, use_relu = True, BN = True,
				             Scope = self.scope + '/decoder/b' + str(i))
		return out


def conv2d(x, kernel, bias, dense = False, use_relu = True, Scope = None, BN = True):
	# padding image with reflection mode
	#x_padded = tf.pad(x, [[0, 0], [1, 1], [1, 1], [0, 0]], mode = 'REFLECT')
	x_padded = tf.pad(x, [[0, 0], [1, 1], [1, 1], [0, 0]], mode='REFLECT')
	# conv and add bias
	out = tf.nn.conv2d(x_padded, kernel, strides = [1, 1, 1, 1], padding = 'VALID')
	out = tf.nn.bias_add(out, bias)
	if BN:
		with tf.compat.v1.variable_scope(Scope):
			out = tf.compat.v1.layers.batch_normalization(out, training = True)
	if use_relu:
		out = tf.nn.leaky_relu(out)
	if dense:
		out = tf.concat([out, x], 3)
	return out

def up_sample(x, scale_factor = 2):
	_, h, w, _ = x.get_shape().as_list()
	new_size = [h * scale_factor, w * scale_factor]
	return tf.compat.v1.image.resize_nearest_neighbor(x, size = new_size)


def channel_attenstion(inputs, ratio=0.25):
	'''ratio代表第一个全连接层下降通道数的倍数'''

	channel = inputs.shape[-1]  # 获取输入特征图的通道数
	channel = int(channel)
	# 分别对输出特征图进行全局最大池化和全局平均池化
	# [h,w,c]==>[None,c]
	x_max = keras.layers.GlobalMaxPooling2D()(inputs)
	x_avg = keras.layers.GlobalAveragePooling2D()(inputs)

	# [None,c]==>[1,1,c]
	x_max = keras.layers.Reshape([1, 1, -1])(x_max)  # -1代表自动寻找通道维度的大小
	x_avg = keras.layers.Reshape([1, 1, -1])(x_avg)  # 也可以用变量channel代替-1

	# 第一个全连接层通道数下降1/4, [1,1,c]==>[1,1,c//4]
	x_max = keras.layers.Dense(channel * ratio)(x_max)
	x_avg = keras.layers.Dense(channel * ratio)(x_avg)

	# relu激活函数
	x_max = keras.layers.Activation('relu')(x_max)
	x_avg = keras.layers.Activation('relu')(x_avg)

	# 第二个全连接层上升通道数, [1,1,c//4]==>[1,1,c]
	x_max = keras.layers.Dense(channel)(x_max)
	x_avg = keras.layers.Dense(channel)(x_avg)

	# 结果在相叠加 [1,1,c]+[1,1,c]==>[1,1,c]
	x = keras.layers.Add()([x_max, x_avg])

	# 经过sigmoid归一化权重
	x = tf.nn.sigmoid(x)

	# 输入特征图和权重向量相乘，给每个通道赋予权重
	x = keras.layers.Multiply()([inputs, x])  # [h,w,c]*[1,1,c]==>[h,w,c]

	return x


# （2）空间注意力机制
def spatial_attention(inputs):
	# 在通道维度上做最大池化和平均池化[b,h,w,c]==>[b,h,w,1]
	# keepdims=Fale那么[b,h,w,c]==>[b,h,w]
	x_max = tf.reduce_max(inputs, axis=3, keepdims=True)  # 在通道维度求最大值
	x_avg = tf.reduce_mean(inputs, axis=3, keepdims=True)  # axis也可以为-1

	# 在通道维度上堆叠[b,h,w,2]
	x = keras.layers.concatenate([x_max, x_avg])

	# 1*1卷积调整通道[b,h,w,1]
	x = keras.layers.Conv2D(filters=1, kernel_size=(1, 1), strides=1, padding='same')(x)

	# sigmoid函数权重归一化
	x = tf.nn.sigmoid(x)

	# 输入特征图和权重相乘
	x = keras.layers.Multiply()([inputs, x])

	return x


# （3）CBAM注意力
def CBAM_attention(inputs):
	# 先经过通道注意力再经过空间注意力
	x = channel_attenstion(inputs)
	x = spatial_attention(x)
	#x = tf.compat.v1.layers.batch_normalization(x, training=True)
	return x


class CondenseNetV2(object):
    def __init__(self, input_channels, growth_rate, num_layers, scope):
        self.scope = scope
        self.weight_vars = []
        with tf.compat.v1.variable_scope(self.scope):
            for i in range(num_layers):
                self.weight_vars.append(self._create_variables(input_channels + i * growth_rate, growth_rate, 3, scope='condense_block_conv{}'.format(i)))

    def _create_variables(self, input_filters, output_filters, kernel_size, scope):
        shape = [kernel_size, kernel_size, input_filters, output_filters]
        with tf.compat.v1.variable_scope(scope):
            kernel = tf.Variable(tf.compat.v1.truncated_normal(shape, stddev=WEIGHT_INIT_STDDEV),
                                 name='kernel')
            bias = tf.Variable(tf.zeros([output_filters]), name='bias')
        return (kernel, bias)

    def forward(self, x):
        for i in range(len(self.weight_vars)):
            kernel, bias = self.weight_vars[i]
            x = conv2d(x, kernel, bias, dense=True, use_relu=True, Scope=self.scope + '/condense_block/b' + str(i))
        return x

def conv_layer(inputs, filters, kernel_size, strides, padding='same', activation=tf.nn.leaky_relu):
    return tf.layers.conv2d(inputs, filters, kernel_size, strides, padding, activation=activation)

def upsample_layer(inputs, factor):
    return tf.keras.layers.UpSampling2D(size=(factor, factor))(inputs)


def fpn(inputs, num_filters, num_feature_maps=1):
	feature_map_channels = inputs.shape[-1] // num_feature_maps
	inputs = [inputs[..., i * feature_map_channels:(i + 1) * feature_map_channels] for i in range(num_feature_maps)]
	pyramid = []
	last_layer = None

	for idx, input_layer in enumerate(reversed(inputs)):
		if last_layer is not None:
			upsampled_layer = upsample_layer(last_layer, 2)
			input_layer = tf.slice(input_layer, [0, 0, 0, 0], [-1, upsampled_layer.shape[1], upsampled_layer.shape[2], -1])
			input_layer = conv_layer(input_layer, upsampled_layer.shape[-1], kernel_size=1, strides=1)
			input_layer = tf.add(input_layer, upsampled_layer)
		last_layer = conv_layer(input_layer, num_filters, kernel_size=3, strides=1)
		if last_layer is not None:
			pyramid.append(last_layer)
		#pyramid.append(last_layer)
	pyramid.reverse()
		# 将输出特征图沿通道维度连接起来，使其与输入格式一致
	output = tf.concat(pyramid, axis=-1)
	return output
