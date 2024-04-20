import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, Dense, GlobalMaxPooling2D
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing import image_dataset_from_directory



# Build the model
def build_model(input_shape=(120, 120, 3)):
    input_layer = Input(shape=input_shape)
    base_model = VGG16(include_top=False, input_tensor=input_layer)
    
    pooled_features = GlobalMaxPooling2D()(base_model.output)
    
    classifier_output = Dense(2048, activation='relu')(pooled_features)
    classifier_output = Dense(1, activation='sigmoid', name='class_output')(classifier_output)
    
    regressor_output = Dense(2048, activation='relu')(pooled_features)
    regressor_output = Dense(4, activation='sigmoid', name='bbox_output')(regressor_output)
    
    model = Model(inputs=input_layer, outputs=[classifier_output, regressor_output])
    return model

# Localization loss
def localization_loss(y_true, yhat):            
    delta_coord = tf.reduce_sum(tf.square(y_true[:,:2] - yhat[:,:2]))
                  
    h_true = y_true[:,3] - y_true[:,1] 
    w_true = y_true[:,2] - y_true[:,0] 

    h_pred = yhat[:,3] - yhat[:,1] 
    w_pred = yhat[:,2] - yhat[:,0] 
    
    delta_size = tf.reduce_sum(tf.square(w_true - w_pred) + tf.square(h_true-h_pred))
    
    return delta_coord + delta_size

# Hard negative mining
def hard_negative_mining(model, negative_images, num_hard_negatives):
    negative_preds = model.predict(negative_images)
    hard_negative_indices = np.argsort(negative_preds[0][:,0])[-num_hard_negatives:]
    hard_negatives = negative_images[hard_negative_indices]
    return hard_negatives

# Train the model with hard negative mining
def train_with_hard_negatives(model, train_data, epochs, negative_images, num_iterations, num_hard_negatives):
    for i in range(num_iterations):
        print(f"Hard negative mining iteration {i+1}/{num_iterations}")
        
        model.fit(train_data, epochs=epochs)
        
        hard_negatives = hard_negative_mining(model, negative_images, num_hard_negatives)
        
        train_data = np.concatenate([train_data, hard_negatives])
        
    return model

# Main training loop
# Main training loop
if __name__ == "__main__":
    facetracker = build_model()
    
    batches_per_epoch = len(train)
    lr_decay = (1. / 0.75 - 1) / batches_per_epoch
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)

    facetracker.compile(optimizer=optimizer,
                        loss={'class_output': tf.keras.losses.BinaryCrossentropy(),
                              'bbox_output': localization_loss},
                        metrics={'class_output': ['accuracy'], 'bbox_output': []})
    
    num_hard_negative_iterations = 3
    num_hard_negatives_per_iteration = 100
    
    # Load negative images
    negative_images = image_dataset_from_directory(
        'path/to/negative/images/directory',
        image_size=(120, 120),
        batch_size=32,
        shuffle=True
    )
    
    trained_model = train_with_hard_negatives(facetracker, 
                                              train, 
                                              epochs=5, 
                                              negative_images=negative_images, 
                                              num_iterations=num_hard_negative_iterations,
                                              num_hard_negatives=num_hard_negatives_per_iteration)
