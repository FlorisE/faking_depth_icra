import tensorflow as tf
import tensorflow_addons as tfa

class InstanceNormalization(tf.keras.layers.Layer):
    def __init__(self, epsilon=1e-5):
        super(InstanceNormalization, self).__init__()
        self.epsilon = epsilon
  
    def build(self, input_shape):
        self.scale = self.add_weight(
            name='scale',
            shape=input_shape[-1:],
            initializer=tf.random_normal_initializer(1., 0.02),
            trainable=True)
  
        self.offset = self.add_weight(
            name='offset',
            shape=input_shape[-1:],
            initializer='zeros',
            trainable=True)
  
    def call(self, x):
        mean, variance = tf.nn.moments(x, axes=[1, 2], keepdims=True)
        inv = tf.math.rsqrt(variance + self.epsilon)
        normalized = (x - mean) * inv
        return self.scale * normalized + self.offset

class SequentialMultiHeadAttention(tf.keras.layers.MultiHeadAttention):
    def __init__(self, num_heads, key_dim):
        super(SequentialMultiHeadAttention, self).__init__(num_heads=num_heads, key_dim=key_dim)
    
    def __call__(self, input_value, training=None):
        return super(SequentialMultiHeadAttention, self).__call__(input_value, input_value)
        
def downsample(filters, size, norm_type='batchnorm', apply_norm=True, name=None, spectral_norm=False, conv_after=False, use_attention=False):
    initializer = tf.random_normal_initializer(0., 0.02)
  
    result = tf.keras.Sequential(name=name)
    
    conv_layer = tf.keras.layers.Conv2D(filters, size, strides=2, padding='same',
                               kernel_initializer=initializer, use_bias=False)
   
    
    if not conv_after:
        if spectral_norm:
            conv_layer = tfa.layers.SpectralNormalization(conv_layer)

        result.add(conv_layer)
        
    if apply_norm:
        if norm_type.lower() == 'batchnorm':
            result.add(tf.keras.layers.BatchNormalization())
        elif norm_type.lower() == 'instancenorm':
            result.add(InstanceNormalization())
        else:
            print("Invalid norm type")
  
    result.add(tf.keras.layers.LeakyReLU())
    
    if use_attention:
        result.add(SequentialMultiHeadAttention(2, 2))
    
    if conv_after:
        if spectral_norm:
            conv_layer = tfa.layers.SpectralNormalization(conv_layer)

        result.add(conv_layer)
  
    return result

def upsample(filters, size, norm_type='batchnorm', dropout_rate=None, name=None, conv_after=False, use_attention=False):
    initializer = tf.random_normal_initializer(0., 0.02)
  
    result = tf.keras.Sequential()
    
    if not conv_after:
        result.add(
            tf.keras.layers.Conv2DTranspose(filters, size, strides=2,
                                            padding='same',
                                            kernel_initializer=initializer,
                                            use_bias=False))
  
    if norm_type.lower() == 'batchnorm':
        result.add(tf.keras.layers.BatchNormalization())
    elif norm_type.lower() == 'instancenorm':
        result.add(InstanceNormalization())
    elif norm_type.lower() == 'spectralnorm':
        result.add(tfa.layers.SpectralNormalization())
  
    if dropout_rate:
        result.add(tf.keras.layers.Dropout(dropout_rate))
  
    result.add(tf.keras.layers.ReLU())
    
    if use_attention:
        result.add(SequentialMultiHeadAttention(2, 2))
    
    if conv_after:
        result.add(
            tf.keras.layers.Conv2DTranspose(filters, size, strides=2,
                                            padding='same',
                                            kernel_initializer=initializer,
                                            use_bias=False))
  
    return result

def uresnet_generator(output_channels, input_channels=1, dropout_rate=0.5, norm_type='batchnorm'):
    initializer = tf.random_normal_initializer(0., 0.02)
    
    inputs = tf.keras.layers.Input(shape=[512, 512, input_channels])
    x = inputs
    
    conv0 = tf.keras.layers.Conv2D(64/2, 3, strides=1, padding='same', kernel_initializer=initializer, use_bias=True)(x)

    def block(x, filters, mode, num_blocks=2):
        for i in range(num_blocks):
            x_in = x
            if norm_type.lower() == 'batchnorm':
                x = tf.keras.layers.BatchNormalization()(x)
            elif norm_type.lower() == 'instancenorm':
                x = InstanceNormalization()(x)
            
            x = tf.keras.layers.LeakyReLU()(x)
            
            x = tf.keras.layers.Conv2D(filters, 3, strides=1, padding='same', kernel_initializer=initializer, use_bias=True)(x)
            
            x_in = tf.keras.layers.Dense(filters)(x_in)
            x = tf.math.divide(tf.math.add(x, x_in), tf.math.sqrt(2.0))
            
        if mode == 'down':
            x = tf.keras.layers.Conv2D(filters, 4, strides=2, padding='same', kernel_initializer=initializer, use_bias=False)(x)
        elif mode == 'up':
            x = tf.keras.layers.Conv2DTranspose(filters, 4, strides=2, padding='same', kernel_initializer=initializer, use_bias=False)(x)
        return x
    
    down1 = block(conv0, 128/2, mode='down')
    down2 = block(down1, 256/2, mode='down')
    down3 = block(down2, 512/2, mode='down')
    down4 = block(down3, 1024/2, mode='down')
    down5 = block(down4, 1024/2, mode='down')
    down6 = block(down5, 1024/2, mode='down')
    down7 = block(down6, 1024/2, mode='down')
    down8 = block(down7, 1024/2, mode='down')
    
    bottom = block(down8, 1024/2, mode='neutral')
    
    concat = tf.keras.layers.Concatenate()
    
    up1 = block(bottom, 1024/2, mode='up')
    skipped1 = concat([down7, up1])
    up2 = block(skipped1, 1024/2, mode='up')
    skipped2 = concat([down6, up2])
    up3 = block(skipped2, 1024/2, mode='up')
    skipped3 = concat([down5, up3])
    up4 = block(skipped3, 1024/2, mode='up')
    skipped4 = concat([down4, up4])
    up5 = block(skipped4, 512/2, mode='up')
    skipped5 = concat([down3, up5])
    up6 = block(skipped5, 256/2, mode='up')
    skipped6 = concat([down2, up6])
    up7 = block(skipped6, 128/2, mode='up')
    skipped7 = concat([down1, up7])
    up8 = block(skipped7, 64/2, mode='up')
    skipped8 = concat([conv0, up8])
    
    last = tf.keras.layers.Conv2D(output_channels, 3, strides=1, padding='same', kernel_initializer=initializer, use_bias=True, activation='tanh')(skipped8)
    
    return tf.keras.Model(inputs=inputs, outputs=last)
    
def unet_generator512(output_channels, input_channels=1, dropout_rate=0.5, norm_type='batchnorm'):
    initializer = tf.random_normal_initializer(0., 0.02)
  
    inputs = tf.keras.layers.Input(shape=[512, 512, input_channels])
    x = inputs
    
    conv0 = tf.keras.layers.Conv2D(64, 4, strides=2, padding='same', kernel_initializer=initializer, use_bias=False)(x)
    relu0 = tf.keras.layers.LeakyReLU()(conv0)
    
    def down(in_val, filters):
        conv = tf.keras.layers.Conv2D(filters, 4, strides=2, padding='same', kernel_initializer=initializer, use_bias=False)(in_val)
        if norm_type.lower() == 'batchnorm':
            norm = tf.keras.layers.BatchNormalization()(conv)
        elif norm_type.lower() == 'instancenorm':
            norm = InstanceNormalization()(conv)
        relu = tf.keras.layers.LeakyReLU()(norm)
        return conv, relu
    
    conv1, relu1 = down(relu0, 128)
    conv2, relu2 = down(relu1, 256)
    conv3, relu3 = down(relu2, 512)
    conv4, relu4 = down(relu3, 1024)
    conv5, relu5 = down(relu4, 1024)
    conv6, relu6 = down(relu5, 1024)
    conv7, relu7 = down(relu6, 1024)
    conv8, relu8 = down(relu7, 1024)
    
    def up(in_val, filters, dropout_rate=None):
        convt = tf.keras.layers.Conv2DTranspose(filters, 4, strides=2, padding='same', kernel_initializer=initializer, use_bias=False)(in_val)
        if norm_type.lower() == 'batchnorm':
            norm = tf.keras.layers.BatchNormalization()(convt)
        elif norm_type.lower() == 'instancenorm':
            norm = InstanceNormalization()(convt)
        if dropout_rate is not None:
            dropout = tf.keras.layers.Dropout(dropout_rate)(norm)
            relu = tf.keras.layers.ReLU()(dropout)
        else:
            relu = tf.keras.layers.ReLU()(norm)
        return convt, relu
    
    concat = tf.keras.layers.Concatenate()
    
    convt1, relu9 = up(relu8, 1024)
    skipped1 = concat([relu7, relu9])
    convt2, relu10 = up(skipped1, 1024)
    skipped2 = concat([relu6, relu10])
    convt3, relu11 = up(skipped2, 1024)
    skipped3 = concat([relu5, relu11])
    convt4, relu12 = up(skipped3, 1024)
    skipped4 = concat([relu4, relu12])
    convt5, relu13 = up(skipped4, 512)
    skipped5 = concat([relu3, relu13])
    convt6, relu14 = up(skipped5, 256)
    skipped6 = concat([relu2, relu14])
    convt7, relu15 = up(skipped6, 128)
    skipped7 = concat([relu1, relu15])
    convt8, relu16 = up(skipped7, 64)
    skipped8 = concat([relu0, relu16])
    
    #last = tf.keras.layers.Conv2DTranspose(output_channels, 4, strides=2, padding='same', kernel_initializer=initializer, activation='tanh')(skipped8)
    last = tf.keras.layers.Conv2DTranspose(output_channels, 4, strides=2, padding='same', kernel_initializer=initializer, activation='tanh')(skipped8)
    
    return tf.keras.Model(inputs=inputs, outputs=last)
    
def unet_generator_backbone(inputs, output_channels, norm_type='batchnorm', size=256, input_channels=1, dropout_rate=0.5, prefix='', conv_after=False, use_attention=False):
    if size == 512:
        n_conv = 1024
    else:
        n_conv = 512
    down_stack = [
        downsample(64, 4, norm_type, apply_norm=False, name=prefix+'down1', conv_after=conv_after, use_attention=False),  # (bs, 128, 128, 64)
        downsample(128, 4, norm_type, name=prefix+'down2', conv_after=conv_after, use_attention=False),  # (bs, 64, 64, 128)
        downsample(256, 4, norm_type, name=prefix+'down3', conv_after=conv_after, use_attention=False),  # (bs, 32, 32, 256)
        downsample(n_conv, 4, norm_type, name=prefix+'down_n_conv_1', conv_after=conv_after, use_attention=use_attention),  # (bs, 16, 16, 512)
        downsample(n_conv, 4, norm_type, name=prefix+'down_n_conv_2', conv_after=conv_after, use_attention=use_attention),  # (bs, 8, 8, 512)
        downsample(n_conv, 4, norm_type, name=prefix+'down_n_conv_3', conv_after=conv_after, use_attention=use_attention),  # (bs, 4, 4, 512)
        downsample(n_conv, 4, norm_type, name=prefix+'down_n_conv_4', conv_after=conv_after, use_attention=use_attention),  # (bs, 2, 2, 512)
        downsample(n_conv, 4, norm_type, name=prefix+'down_n_conv_5', conv_after=conv_after, use_attention=use_attention),  # (bs, 1, 1, 512)
    ]
    
    up_stack = [
        upsample(n_conv, 4, norm_type, dropout_rate, name=prefix+'up1', conv_after=conv_after, use_attention=use_attention),  # (bs, 2, 2, 1024)
        upsample(n_conv, 4, norm_type, dropout_rate, name=prefix+'up2', conv_after=conv_after, use_attention=use_attention),  # (bs, 4, 4, 1024)
        upsample(n_conv, 4, norm_type, dropout_rate, name=prefix+'up3', conv_after=conv_after, use_attention=use_attention),  # (bs, 8, 8, 1024)
        upsample(n_conv, 4, norm_type, name=prefix+'up4', conv_after=conv_after, use_attention=use_attention),  # (bs, 16, 16, 1024)
        upsample(256, 4, norm_type, name=prefix+'up256', conv_after=conv_after, use_attention=False),  # (bs, 32, 32, 512)
        upsample(128, 4, norm_type, name=prefix+'up128', conv_after=conv_after, use_attention=False),  # (bs, 64, 64, 256)
        upsample(64, 4, norm_type, name=prefix+'up64', conv_after=conv_after, use_attention=False),  # (bs, 128, 128, 128)
    ]
        
    if size == 512:
        down_stack.insert(3, downsample(512, 4, norm_type, name=prefix+'down4', conv_after=conv_after, use_attention=use_attention))
        up_stack.insert(4, upsample(512, 4, norm_type, dropout_rate, name=prefix+'up5', conv_after=conv_after, use_attention=use_attention))
  
    initializer = tf.random_normal_initializer(0., 0.02)
    last = tf.keras.layers.Conv2DTranspose(
        output_channels, 4, strides=2,
        padding='same', kernel_initializer=initializer,
        activation='tanh')  # (bs, 256, 256, 3)
  
    concat = tf.keras.layers.Concatenate()
    
    x = inputs
  
    # Downsampling through the model
    skips = []
    for down in down_stack:
        x = down(x)
        skips.append(x)
  
    skips = reversed(skips[:-1])
  
    # Upsampling and establishing the skip connections
    for up, skip in zip(up_stack, skips):
        x = up(x)
        x = concat([x, skip])
  
    x = last(x)
    
    return x
    
def unet_generator(output_channels, norm_type='batchnorm', size=256, input_channels=1, dropout_rate=0.5, conv_after=False, use_attention=False):
    inputs = tf.keras.layers.Input(shape=[size, size, input_channels])
    outputs = unet_generator_backbone(inputs, output_channels, norm_type, size, input_channels, dropout_rate, conv_after=conv_after, use_attention=use_attention)
  
    return tf.keras.Model(inputs=inputs, outputs=outputs)

def unet_generator2(output_channels, norm_type='batchnorm', size=256, input_channels=1, dropout_rate=0.5):
    if size == 512:
        n_conv = 2048
    else:
        n_conv = 1024
    down_stack = [
        downsample(128, 4, norm_type, apply_norm=False),  # (bs, 128, 128, 64)
        downsample(256, 4, norm_type),  # (bs, 64, 64, 128)
        downsample(512, 4, norm_type),  # (bs, 32, 32, 256)
        downsample(n_conv, 4, norm_type),  # (bs, 16, 16, 512)
        downsample(n_conv, 4, norm_type),  # (bs, 8, 8, 512)
        downsample(n_conv, 4, norm_type),  # (bs, 4, 4, 512)
        downsample(n_conv, 4, norm_type),  # (bs, 2, 2, 512)
        downsample(n_conv, 4, norm_type),  # (bs, 1, 1, 512)
    ]
    
    up_stack = [
        upsample(n_conv, 4, norm_type, dropout_rate),  # (bs, 2, 2, 1024)
        upsample(n_conv, 4, norm_type, dropout_rate),  # (bs, 4, 4, 1024)
        upsample(n_conv, 4, norm_type, dropout_rate),  # (bs, 8, 8, 1024)
        upsample(n_conv, 4, norm_type),  # (bs, 16, 16, 1024)
        upsample(256, 4, norm_type),  # (bs, 32, 32, 512)
        upsample(128, 4, norm_type),  # (bs, 64, 64, 256)
        upsample(64, 4, norm_type),  # (bs, 128, 128, 128)
    ]
        
    if size == 512:
        down_stack.insert(3, downsample(512, 4, norm_type))
        up_stack.insert(4, upsample(512, 4, norm_type, dropout_rate))
  
    initializer = tf.random_normal_initializer(0., 0.02)
    last = tf.keras.layers.Conv2DTranspose(
        output_channels, 4, strides=2,
        padding='same', kernel_initializer=initializer,
        activation='tanh')  # (bs, 256, 256, 3)
  
    concat = tf.keras.layers.Concatenate()
  
    inputs = tf.keras.layers.Input(shape=[size, size, input_channels])
    x = inputs
  
    # Downsampling through the model
    skips = []
    for down in down_stack:
        x = down(x)
        skips.append(x)
  
    skips = reversed(skips[:-1])
  
    # Upsampling and establishing the skip connections
    for up, skip in zip(up_stack, skips):
        x = up(x)
        x = concat([x, skip])
  
    x = last(x)
  
    return tf.keras.Model(inputs=inputs, outputs=x)

def unet_discriminator(norm_type='batchnorm', target=True, size=256, channels=1, spectral_norm=False):
    initializer = tf.random_normal_initializer(0., 0.02)
  
    inp = tf.keras.layers.Input(shape=[size, size, channels], name='input_image')
    x = inp
  
    if target:
        tar = tf.keras.layers.Input(shape=[size, size, channels], name='target_image')
        x = tf.keras.layers.concatenate([inp, tar])  # (bs, 256, 256, channels*2)
  
    down1 = downsample(64, 4, norm_type, False, spectral_norm=spectral_norm)(x)  # (bs, 128, 128, 64)
    down2 = downsample(128, 4, norm_type, spectral_norm=spectral_norm)(down1)  # (bs, 64, 64, 128)
    down3 = downsample(256, 4, norm_type, spectral_norm=spectral_norm)(down2)  # (bs, 32, 32, 256)
    
    if size == 512:
        down4 = downsample(512, 4, norm_type, spectral_norm=spectral_norm)(down3)
        last = down4
        n_conv = 1024
    else:
        last = down3
        n_conv = 512
  
    zero_pad1 = tf.keras.layers.ZeroPadding2D()(last)  # (bs, 34, 34, 256)
    
    conv_layer = tf.keras.layers.Conv2D(n_conv, 4, strides=1, kernel_initializer=initializer, use_bias=False)
    
    if spectral_norm:
        conv_layer = tfa.layers.SpectralNormalization(conv_layer)

    if norm_type.lower() == 'batchnorm':
        norm1 = tf.keras.layers.BatchNormalization()(conv_layer(zero_pad1))
    elif norm_type.lower() == 'instancenorm':
        norm1 = InstanceNormalization()(conv_layer(zero_pad1))
    else:
        print("Invalid norm type")
  
    leaky_relu = tf.keras.layers.LeakyReLU(alpha=0.2)(norm1)
    zero_pad2 = tf.keras.layers.ZeroPadding2D()(leaky_relu)  # (bs, 33, 33, 512)
  
    last = tf.keras.layers.Conv2D(
        1, 4, strides=1,
        kernel_initializer=initializer)(zero_pad2)  # (bs, 30, 30, 1)
  
    if target:
        return tf.keras.Model(inputs=[inp, tar], outputs=last)
    else:
        return tf.keras.Model(inputs=inp, outputs=last)


def conv_discriminator(norm_type='batchnorm', size=256, channels=1, spectral_norm=False):
    initializer = tf.random_normal_initializer(0., 0.02)
  
    inp = tf.keras.layers.Input(shape=[size, size, channels], name='input_image')
    x = inp
    
    down = downsample(64, 4, norm_type, spectral_norm=spectral_norm)(x)
    down = downsample(128, 4, norm_type, spectral_norm=spectral_norm)(down)
    down = downsample(256, 4, norm_type, spectral_norm=spectral_norm)(down)
    down = downsample(512, 4, norm_type, spectral_norm=spectral_norm)(down)
    down = downsample(1024, 4, norm_type, spectral_norm=spectral_norm)(down)
    
    if size == 512:
        last_downsample = downsample(1024, 4, norm_type, spectral_norm=spectral_norm)(down)
    else:
        last_downsample = down
    
    conv_layer = tf.keras.layers.Conv2D(
        1, 1, strides=1,
        kernel_initializer=initializer)
    
    if spectral_norm:
        conv_layer = tfa.layers.SpectralNormalization(conv_layer)
    
    last_conv = conv_layer(last_downsample)
    
    last_flatten = tf.keras.layers.Flatten()(last_conv)
    
    last_dense = tf.keras.layers.Dense(1)(last_flatten)
    
    return tf.keras.Model(inputs=inp, outputs=last_dense)
    
    
# generator a resnet block
def resnet_block(n_filters, input_layer):
    # weight initialization
    init = tf.keras.initializers.RandomNormal(stddev=0.02)
    # first layer convolutional layer
    padding = tf.constant([[0, 0], [1, 1], [1, 1], [0, 0]])
    g = tf.keras.layers.Lambda(lambda x: tf.pad(x, padding, "REFLECT"))(input_layer)
    g = tf.keras.layers.Conv2D(n_filters, (3,3), padding='valid', kernel_initializer=init)(g)
    g = tfa.layers.InstanceNormalization(axis=-1)(g)
    g = tf.keras.layers.Activation('relu')(g)
    # second convolutional layer
    g = tf.keras.layers.Lambda(lambda x: tf.pad(x, padding, "REFLECT"))(g)
    g = tf.keras.layers.Conv2D(n_filters, (3,3), padding='valid', kernel_initializer=init)(g)
    g = tfa.layers.InstanceNormalization(axis=-1)(g)
    # concatenate merge channel-wise with input layer
    g = tf.keras.layers.Concatenate()([g, input_layer])
    return g

def resnet_generator(n_resnet=9, size=256, input_channels=1, output_channels=1):
    # weight initialization
    init = tf.keras.initializers.RandomNormal(stddev=0.02)
    # image input
    in_image = tf.keras.layers.Input(shape=[size, size, input_channels])
    # c7s1-64
    g = tf.keras.layers.Conv2D(64, (7,7), padding='same', kernel_initializer=init)(in_image)
    g = tfa.layers.InstanceNormalization(axis=-1)(g)
    g = tf.keras.layers.Activation('relu')(g)
    # d128
    g = tf.keras.layers.Conv2D(128, (3,3), strides=(2,2), padding='same', kernel_initializer=init)(g)
    g = tfa.layers.InstanceNormalization(axis=-1)(g)
    g = tf.keras.layers.Activation('relu')(g)
    # d256
    g = tf.keras.layers.Conv2D(256, (3,3), strides=(2,2), padding='same', kernel_initializer=init)(g)
    g = tfa.layers.InstanceNormalization(axis=-1)(g)
    g = tf.keras.layers.Activation('relu')(g)
    # R256
    for _ in range(n_resnet):
        g = resnet_block(256, g)
    # u128
    g = tf.keras.layers.Conv2DTranspose(128, (3,3), strides=(2,2), padding='same', kernel_initializer=init)(g)
    g = tfa.layers.InstanceNormalization(axis=-1)(g)
    g = tf.keras.layers.Activation('relu')(g)
    # u64
    g = tf.keras.layers.Conv2DTranspose(64, (3,3), strides=(2,2), padding='same', kernel_initializer=init)(g)
    g = tfa.layers.InstanceNormalization(axis=-1)(g)
    g = tf.keras.layers.Activation('relu')(g)
    # c7s1-3
    g = tf.keras.layers.Conv2D(output_channels, (7,7), padding='same', kernel_initializer=init)(g)
    g = tfa.layers.InstanceNormalization(axis=-1)(g)
    out_image = tf.keras.layers.Activation('tanh')(g)
    # define model
    model = tf.keras.models.Model(in_image, out_image)
    return model

loss_obj = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def discriminator_loss_minimax(real, generated):
    real_loss = loss_obj(tf.ones_like(real), real)

    generated_loss = loss_obj(tf.zeros_like(generated), generated)

    total_disc_loss = real_loss + generated_loss

    return total_disc_loss * 0.5

def generator_loss_minimax(generated):
    return loss_obj(tf.ones_like(generated), generated)

def discriminator_loss_wasserstein(real, fake):
    return tf.reduce_mean(tf.abs(real - fake))

def generator_loss_wasserstein(critiqued_fake):
    return tf.reduce_mean(critiqued_fake)

def calc_cycle_loss(real_image, cycled_image):
    loss1 = tf.reduce_mean(tf.abs(real_image - cycled_image))

    return loss1

def identity_loss(real_image, same_image):
    loss = tf.reduce_mean(tf.abs(real_image - same_image))
    return 0.5 * loss

@tf.function
def train_step(real_x,
               real_y,
               generator_g,
               generator_f,
               discriminator_x,
               discriminator_y,
               generator_g_optimizer,
               generator_f_optimizer,
               discriminator_x_optimizer,
               discriminator_y_optimizer):
    # persistent is set to True because the tape is used more than
    # once to calculate the gradients.
    with tf.GradientTape(persistent=True) as tape:
        # Generator G translates X -> Y
        # Generator F translates Y -> X.
        
        fake_y = generator_g(real_x, training=True)
        cycled_x = generator_f(fake_y, training=True)
    
        fake_x = generator_f(real_y, training=True)
        cycled_y = generator_g(fake_x, training=True)
    
        # same_x and same_y are used for identity loss.
        same_x = generator_f(real_x, training=True)
        same_y = generator_g(real_y, training=True)
    
        disc_real_x = discriminator_x(real_x, training=True)
        disc_real_y = discriminator_y(real_y, training=True)
    
        disc_fake_x = discriminator_x(fake_x, training=True)
        disc_fake_y = discriminator_y(fake_y, training=True)
    
        # calculate the loss
        gen_g_loss = generator_loss_minimax(disc_fake_y)
        gen_f_loss = generator_loss_minimax(disc_fake_x)
        #gen_g_loss = generator_loss_wasserstein(disc_fake_y)
        #gen_f_loss = generator_loss_wasserstein(disc_fake_x)
    
        total_cycle_loss = calc_cycle_loss(real_x, cycled_x) + calc_cycle_loss(real_y, cycled_y)
    
        # Total generator loss = adversarial loss + cycle loss
        total_gen_g_loss = gen_g_loss + total_cycle_loss + identity_loss(real_y, same_y)
        total_gen_f_loss = gen_f_loss + total_cycle_loss + identity_loss(real_x, same_x)
    
        disc_x_loss = discriminator_loss_minimax(disc_real_x, disc_fake_x)
        disc_y_loss = discriminator_loss_minimax(disc_real_y, disc_fake_y)
        #disc_x_loss = discriminator_loss_wasserstein(disc_real_x, disc_fake_x)
        #disc_y_loss = discriminator_loss_wasserstein(disc_real_y, disc_fake_y)

    # Calculate the gradients for generator and discriminator
    generator_g_gradients = tape.gradient(total_gen_g_loss, 
                                          generator_g.trainable_variables)
    generator_f_gradients = tape.gradient(total_gen_f_loss, 
                                          generator_f.trainable_variables)
  
    discriminator_x_gradients = tape.gradient(disc_x_loss, 
                                              discriminator_x.trainable_variables)
    discriminator_y_gradients = tape.gradient(disc_y_loss, 
                                              discriminator_y.trainable_variables)
  
    # Apply the gradients to the optimizer
    generator_g_optimizer.apply_gradients(zip(generator_g_gradients, 
                                              generator_g.trainable_variables))
  
    generator_f_optimizer.apply_gradients(zip(generator_f_gradients, 
                                              generator_f.trainable_variables))
  
    discriminator_x_optimizer.apply_gradients(zip(discriminator_x_gradients,
                                                  discriminator_x.trainable_variables))
  
    discriminator_y_optimizer.apply_gradients(zip(discriminator_y_gradients,
                                                  discriminator_y.trainable_variables))
    
    return gen_g_loss, gen_f_loss, total_cycle_loss, total_gen_g_loss, total_gen_f_loss, disc_x_loss, disc_y_loss

alpha_0 = 1.0  # generator loss
alpha_1 = 10.0  # total cycle loss
alpha_2 = 1.0  # supercycle loss
alpha_3 = 1.0  # same result loss
alpha_4 = 1.0
alpha_5 = 1.0  # half cycle loss

@tf.function
def train_generator_color_depth(transparent_color,
                                opaque_depth,
                                generator_tc2od,
                                generator_od2tc,
                                generator_tc2od_optimizer,
                                generator_od2tc_optimizer,
                                discriminator_tc,
                                discriminator_od):
    with tf.GradientTape(persistent=True) as tape:
        generated_tc2od = generator_tc2od(transparent_color, training=True)
        cycled_tc2od2tc = generator_od2tc(generated_tc2od, training=True)
                                
        generated_od2tc = generator_od2tc(opaque_depth, training=True)
        cycled_od2tc2od = generator_tc2od(generated_od2tc, training=True)
    
        disc_generated_od2tc = discriminator_tc(generated_od2tc, training=False)
        disc_generated_tc2od = discriminator_od(generated_tc2od, training=False)
        
        gen_tc2od_loss = generator_loss_minimax(disc_generated_tc2od)
        gen_od2tc_loss = generator_loss_minimax(disc_generated_od2tc)
        
        total_cycle_loss = calc_cycle_loss(transparent_color, cycled_tc2od2tc) + calc_cycle_loss(opaque_depth, cycled_od2tc2od)
        
        total_gen_tc2od_loss = alpha_0 * gen_tc2od_loss + alpha_1 * total_cycle_loss # + alpha_2 * total_supercycle_loss + alpha_3 * same_result_loss + alpha_5 * half_cycle_loss_td2tg + alpha_5 * half_cycle_loss_tg2td #  + identity_loss(real_y, same_y)
        total_gen_od2tc_loss = alpha_0 * gen_od2tc_loss
        total_gen_od2tc_loss += alpha_1 * total_cycle_loss
        # total_gen_od2tc_loss += alpha_2 * total_supercycle_loss
        # total_gen_od2tc_loss += alpha_3 * same_result_loss
        # total_gen_od2tc_loss += alpha_5 * half_cycle_loss_td2tg
        # total_gen_od2tc_loss += alpha_5 * half_cycle_loss_tg2td
        # total_gen_od2tc_loss += identity_loss(real_x, same_x)
        
    generator_tc2od_gradients = tape.gradient(total_gen_tc2od_loss, 
                                          generator_tc2od.trainable_variables)
    generator_od2tc_gradients = tape.gradient(total_gen_od2tc_loss, 
                                          generator_od2tc.trainable_variables)
    
    generator_tc2od_optimizer.apply_gradients(zip(generator_tc2od_gradients, 
                                              generator_tc2od.trainable_variables))
    generator_od2tc_optimizer.apply_gradients(zip(generator_od2tc_gradients, 
                                              generator_od2tc.trainable_variables))
    
    return alpha_0 * gen_tc2od_loss, \
           alpha_0 * gen_od2tc_loss, \
           alpha_1 * total_cycle_loss, \
           total_gen_tc2od_loss, \
           total_gen_od2tc_loss, \
           generated_od2tc, \
           generated_tc2od
           #alpha_3 * same_result_loss, \
           #alpha_2 * total_supercycle_loss, \
           #alpha_4 * identity_loss_od2td, \
           #alpha_4 * identity_loss_td2od, \
           #alpha_5 * half_cycle_loss_td2tg, \
           #alpha_5 * half_cycle_loss_tg2td, \

@tf.function
def train_generator_depth_depth(opaque_depth,
                                transparent_depth,
                                generator_od2td,
                                generator_td2od,
                                generator_od2td_optimizer,
                                generator_td2od_optimizer,
                                discriminator_od,
                                discriminator_td):
    with tf.GradientTape(persistent=True) as tape:
        generated_td2od = generator_td2od(transparent_depth, training=True)
        cycled_td2od2td = generator_od2td(generated_td2od, training=True)
        
        generated_od2td = generator_od2td(opaque_depth, training=True)
        cycled_od2td2od = generator_td2od(generated_od2td, training=True)
        
        # generated_tc2od2td = generator_od2td(
        #     generator_tc2od(
        #         transparent_color,
        #         training=True),
        #     training=True)
        # 
        # supercycle_tc2tc = generator_od2tc(
        #     generator_td2od(
        #         generated_tc2od2td,
        #         training=True),
        #     training=True)
        # 
        # generated_td2od2tc = generator_od2tc(
        #     generator_td2od(
        #         transparent_depth,
        #         training=True),
        #     training=True)
        # 
        # supercycle_td2td = generator_od2td(
        #     generator_tc2od(
        #         generated_td2od2tc,
        #         training=True),
        #     training=True)
    
        same_od = generator_td2od(opaque_depth, training=True)
        same_td = generator_od2td(transparent_depth, training=True)
        
        disc_generated_td2od = discriminator_od(generated_td2od, training=False)
        disc_generated_od2td = discriminator_td(generated_od2td, training=False)

        gen_od2td_loss = generator_loss_minimax(disc_generated_od2td)
        gen_td2od_loss = generator_loss_minimax(disc_generated_td2od)
        # gen_g_loss = generator_loss_wasserstein(disc_fake_y)
        # gen_f_loss = generator_loss_wasserstein(disc_fake_x)
        # gen_od2td_loss = generator_loss_wasserstein(disc_generated_td)
        # gen_td2od_loss = generator_loss_wasserstein(disc_generated_od)

        total_cycle_loss = calc_cycle_loss(opaque_depth, cycled_od2td2od) + calc_cycle_loss(transparent_depth, cycled_td2od2td)
        #total_supercycle_loss = calc_cycle_loss(supercycle_td, transparent_depth) + calc_cycle_loss(supercycle_x, real_x)
        
        #same_result_loss = identity_loss(fake_y, generated_transparent_depth)
        
        #half_cycle_loss_td2tg = tf.reduce_mean(tf.abs(transparent_depth - x_g_od2td))
        #half_cycle_loss_tg2td = tf.reduce_mean(tf.abs(real_x - td_f))
        
        identity_loss_od2td = identity_loss(opaque_depth, same_od)
        identity_loss_td2od = identity_loss(transparent_depth, same_td)
        
        total_gen_od2td_loss = alpha_0 * gen_od2td_loss
        total_gen_od2td_loss += alpha_1 * total_cycle_loss
        # total_gen_od2td_loss += alpha_2 * total_supercycle_loss
        # total_gen_od2td_loss += alpha_3 * same_result_loss
        total_gen_od2td_loss += alpha_4 * identity_loss_od2td
        # total_gen_od2td_loss += alpha_5 * half_cycle_loss_td2tg
        # total_gen_od2td_loss += alpha_5 * half_cycle_loss_tg2td
        
        total_gen_td2od_loss = alpha_0 * gen_td2od_loss
        total_gen_td2od_loss += alpha_1 * total_cycle_loss
        # total_gen_td2od_loss += alpha_2 * total_supercycle_loss
        # total_gen_td2od_loss += alpha_3 * same_result_loss
        total_gen_td2od_loss += alpha_4 * identity_loss_td2od
        # total_gen_td2od_loss += alpha_5 * half_cycle_loss_td2tg
        # total_gen_td2od_loss += alpha_5 * half_cycle_loss_tg2td
        
    generator_od2td_gradients = tape.gradient(total_gen_od2td_loss, 
                                          generator_od2td.trainable_variables)
    generator_td2od_gradients = tape.gradient(total_gen_td2od_loss, 
                                          generator_td2od.trainable_variables)
    
    generator_od2td_optimizer.apply_gradients(zip(generator_od2td_gradients, 
                                              generator_od2td.trainable_variables))
    generator_td2od_optimizer.apply_gradients(zip(generator_td2od_gradients, 
                                              generator_td2od.trainable_variables))
    
    return alpha_1 * total_cycle_loss, \
           alpha_0 * gen_od2td_loss, \
           alpha_0 * gen_td2od_loss, \
           total_gen_od2td_loss, \
           total_gen_td2od_loss, \
           generated_td2od, \
           generated_od2td
           #alpha_3 * same_result_loss, \
           #alpha_2 * total_supercycle_loss, \
           #alpha_4 * identity_loss_od2td, \
           #alpha_4 * identity_loss_td2od, \
           #alpha_5 * half_cycle_loss_td2tg, \
           #alpha_5 * half_cycle_loss_tg2td, \

@tf.function
def train_discriminator_color_depth(transparent_color,
                                    generated_od2tc,
                                    opaque_depth,
                                    generated_tc2od,
                                    discriminator_tc,
                                    discriminator_od,
                                    discriminator_tc_optimizer,
                                    discriminator_od_optimizer):
    with tf.GradientTape(persistent=True) as tape:
        disc_transparent_color = discriminator_tc(transparent_color, training=True)
        disc_opaque_depth = discriminator_od(opaque_depth, training=True)
        
        disc_generated_od2tc = discriminator_tc(generated_od2tc, training=True)
        disc_generated_tc2od = discriminator_od(generated_tc2od, training=True)
        
        disc_tc_loss = discriminator_loss_minimax(disc_transparent_color, disc_generated_od2tc)
        disc_od_loss = discriminator_loss_minimax(disc_opaque_depth, disc_generated_tc2od)

    discriminator_tc_gradients = tape.gradient(disc_tc_loss, 
                                               discriminator_tc.trainable_variables)
    discriminator_od_gradients = tape.gradient(disc_od_loss, 
                                              discriminator_od.trainable_variables)
    
    discriminator_tc_optimizer.apply_gradients(zip(discriminator_tc_gradients,
                                                  discriminator_tc.trainable_variables))
    discriminator_od_optimizer.apply_gradients(zip(discriminator_od_gradients,
                                                  discriminator_od.trainable_variables))
    
    return disc_tc_loss, disc_od_loss
        
    
@tf.function
def train_discriminator_depth_depth(opaque_depth,
                                    transparent_depth,
                                    generated_od2td,
                                    generated_td2od,
                                    discriminator_od,
                                    discriminator_td,
                                    discriminator_od_optimizer,
                                    discriminator_td_optimizer):
    with tf.GradientTape(persistent=True) as tape:
        disc_opaque_depth = discriminator_od(opaque_depth, training=True)
        disc_transparent_depth = discriminator_td(transparent_depth, training=True)
    
        disc_generated_td2od = discriminator_od(generated_td2od, training=True)
        disc_generated_od2td = discriminator_td(generated_od2td, training=True)
    
        disc_od_loss = discriminator_loss_minimax(disc_opaque_depth, disc_generated_td2od)
        disc_td_loss = discriminator_loss_minimax(disc_transparent_depth, disc_generated_od2td)
  
    discriminator_od_gradients = tape.gradient(disc_od_loss, 
                                              discriminator_od.trainable_variables)
    discriminator_td_gradients = tape.gradient(disc_td_loss, 
                                              discriminator_td.trainable_variables)
    
    discriminator_od_optimizer.apply_gradients(zip(discriminator_od_gradients,
                                                  discriminator_od.trainable_variables))
    
    discriminator_td_optimizer.apply_gradients(zip(discriminator_td_gradients,
                                                  discriminator_td.trainable_variables))
    
    return disc_od_loss, disc_td_loss