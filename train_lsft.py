from data_loader.dataprocess import load_all_data,process_dataset
from data_loader.dataloader import DataGen
from architecture.arch_loader_lsft import discriminator,gan,generator
import tensorflow as tf
from tensorflow.keras.optimizers import Adam

def compute_contrastive(gen_outputs, labels):
    # Simple contrastive loss: margin loss for same vs different class
    gen_flat = tf.reshape(gen_outputs, (tf.shape(gen_outputs)[0], -1))
    norm = tf.norm(gen_flat, axis=1, keepdims=True)
    gen_norm = gen_flat / (norm + 1e-8)
    sim = tf.matmul(gen_norm, gen_norm, transpose_b=True)
    loss = 0.0
    for i in range(tf.shape(labels)[0]):
        same_class = tf.equal(labels, labels[i])
        pos_mask = tf.logical_and(same_class, tf.not_equal(tf.range(tf.shape(labels)[0]), i))
        neg_mask = tf.logical_not(same_class)
        if tf.reduce_any(pos_mask):
            pos_sim = tf.reduce_max(tf.boolean_mask(sim[i], pos_mask))
            neg_sim = tf.reduce_max(tf.boolean_mask(sim[i], neg_mask))
            loss += tf.maximum(0.0, neg_sim - pos_sim + 0.1)
    return loss / tf.cast(tf.shape(labels)[0], tf.float32)

def train(gener,d_model, g_model,n_epochs=200, n_batch=1):
    #step 1 train the discriminator on Real Images
    #Step 1.1 Get Real Images Batch
    print('LOADING DATA')
    train_data,test_data=load_all_data()
    print('PROCESSING DATA')
    inp_images_train,features_train,out_images_train,y_train= process_dataset(train_data)
    gen=DataGen(n_batch,inp_images_train,features_train,out_images_train)

    steps = inp_images_train.shape[0]/n_batch
    for epoch in range(n_epochs):
        print('Epoch = ',epoch)
        print("Steps = ",steps)
        for step in range(int(steps)):

            inp_batch,feature_batch,out_batch,out_batch_2,y_real = gen.real_batch()
            fake_batch,y_fake=gen.gen_batch(gener)
            gen.update_batch()
            if step %100 == 0:
                d_loss1=d_model.train_on_batch([inp_batch,out_batch],y_real)
                d_loss2 = d_model.train_on_batch([inp_batch,fake_batch],y_fake)
            # Generator training with adversarial + contrastive loss
            with tf.GradientTape() as tape:
                gen_out = gener([inp_batch,feature_batch])
                disc_out = d_model([inp_batch, gen_out])
                adv_loss = tf.reduce_mean(tf.keras.losses.binary_crossentropy(y_real, disc_out))
                contrastive_loss = compute_contrastive(gen_out, y_real)
                g_loss = adv_loss + contrastive_loss
            grads = tape.gradient(g_loss, gener.trainable_variables)
            gen_opt.apply_gradients(zip(grads, gener.trainable_variables))
            print('>%d, d1[%.3f] d2[%.3f] g[%.3f] c[%.3f]' % (step, d_loss1, d_loss2, g_loss, contrastive_loss))
        import os
        os.makedirs(f'models/LSFT/discriminator/{epoch}', exist_ok=True)
        os.makedirs(f'models/LSFT/gan/{epoch}', exist_ok=True)
        os.makedirs(f'models/LSFT/gen/{epoch}', exist_ok=True)
        d_model.save_weights(f'models/LSFT/discriminator/{epoch}/disc')
        g_model.save_weights(f'models/LSFT/gan/{epoch}/gan')
        gener.save_weights(f'models/LSFT/gen/{epoch}/gan')

    return gener

gen_model = generator()
disc = discriminator()
gan_model = gan(gen_model,disc)
gen_opt = Adam(learning_rate=0.0002, beta_1=0.5)

g_model=train(gen_model,disc,gan_model,n_epochs=25,n_batch=32)
