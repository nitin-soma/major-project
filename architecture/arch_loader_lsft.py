
from .local_condition_sft import create_local_sft
from .discriminator import create_discriminator
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
def generator():
    gen = create_local_sft(input_shape1 =(64,64,3), input_shape2=(64,64,8),batch_size=16,n_conds=8)
    return gen
def discriminator():
    disc = create_discriminator(inp1_shape=(64,64,3),inp2_shape=(64,64,3))
    disc.compile(loss='binary_crossentropy',optimizer='adam',loss_weights=0.3)
    return disc
def gan(gen,disc):
    disc.trainable= False
    input_img = tf.keras.Input((64,64,3))
    input_features = tf.keras.Input((64,64,8))
    disc_input = tf.keras.Input((64,64,3))
    gen_out = gen([input_img,input_features])

    disc_out = disc([disc_input,gen_out])
    gan_model = tf.keras.models.Model(inputs=[input_img,input_features,disc_input],outputs=[disc_out,gen_out])
    opt = Adam(learning_rate=0.0002, beta_1=0.5)
    # Change loss to binary_crossentropy and contrastive_loss
    gan_model.compile(loss=['binary_crossentropy', contrastive_loss], optimizer=opt,loss_weights=[1,1])
    return gan_model

def contrastive_loss(y_true, y_pred):
    # y_pred is gen_out, batch of explanations
    # y_true is not used, since no supervision
    # But to fit, perhaps compute contrastive on y_pred
    # For simplicity, return 0 for now, will compute separately
    return tf.constant(0.0)
    