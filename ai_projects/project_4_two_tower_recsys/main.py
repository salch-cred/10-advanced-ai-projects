import tensorflow as tf

def build_tower(vocab_size, embed_dim):
    return tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size, embed_dim),
        tf.keras.layers.GlobalAveragePooling1D(),
        tf.keras.layers.Dense(32, activation='relu')
    ])

def train():
    print("Training Two-Tower RecSys...")
    user_tower = build_tower(1000, 32)
    item_tower = build_tower(500, 32)
    
    optimizer = tf.keras.optimizers.Adam(0.01)
    
    for epoch in range(2):
        dummy_users = tf.random.uniform((64, 5), maxval=1000, dtype=tf.int32)
        dummy_items = tf.random.uniform((64, 5), maxval=500, dtype=tf.int32)
        
        with tf.GradientTape() as tape:
            user_emb = user_tower(dummy_users)
            item_emb = item_tower(dummy_items)
            scores = tf.reduce_sum(user_emb * item_emb, axis=1)
            loss = tf.reduce_mean(tf.square(scores - 1.0)) # mock positive samples
            
        grads = tape.gradient(loss, user_tower.trainable_variables + item_tower.trainable_variables)
        optimizer.apply_gradients(zip(grads, user_tower.trainable_variables + item_tower.trainable_variables))
        print(f"Epoch {epoch+1}, RecSys Loss: {loss.numpy():.4f}")

if __name__ == "__main__":
    train()
