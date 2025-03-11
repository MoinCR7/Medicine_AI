import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences # type: ignore
import os

# Custom layer needed for model loading
class WeightedEmbedding(tf.keras.layers.Layer):
    """
    Custom embedding layer that scales each token's embedding vector
    by its corresponding severity weight.
    """
    def __init__(self, input_dim, output_dim, embedding_scaling, input_length, **kwargs):
        super(WeightedEmbedding, self).__init__(**kwargs)
        self.input_dim = input_dim        # Vocabulary size (+1 for padding)
        self.output_dim = output_dim      # Embedding dimension
        self.input_length = input_length
        
        # If embedding_scaling is not already a list or numpy array,
        # try to convert it to a Python list for serialization.
        if isinstance(embedding_scaling, tf.Tensor):
            self.embedding_scaling = embedding_scaling.numpy().tolist()
        else:
            self.embedding_scaling = embedding_scaling

        # Standard Keras embedding layer
        self.embedding = tf.keras.layers.Embedding(
            input_dim=self.input_dim,
            output_dim=self.output_dim,
            input_length=self.input_length,
            mask_zero=False
        )
    
    def call(self, inputs):
        # Force the inputs to int32 (needed for tf.gather)
        inputs = tf.cast(inputs, tf.int32)
        
        # Get embeddings for the input tokens; shape: (batch_size, sequence_length, output_dim)
        embeddings = self.embedding(inputs)
        
        # Gather the severity weights for each token; shape: (batch_size, sequence_length)
        # We need to convert our stored list back to a tensor.
        scaling_tensor = tf.convert_to_tensor(self.embedding_scaling, dtype=tf.float32)
        weights = tf.gather(scaling_tensor, inputs)
        
        # Expand the last dimension so that weights shape becomes (batch_size, sequence_length, 1)
        weights = tf.expand_dims(weights, axis=-1)
        
        # Multiply each embedding vector by its severity weight
        weighted_embeddings = embeddings * weights
        return weighted_embeddings
    
    def compute_output_shape(self, input_shape):
        # The output shape is (batch_size, sequence_length, output_dim)
        return input_shape + (self.output_dim,)
    
    def get_config(self):
        # Get the base config from the parent class.
        config = super(WeightedEmbedding, self).get_config()
        # Update the config with the parameters of this layer.
        config.update({
            'input_dim': self.input_dim,
            'output_dim': self.output_dim,
            'embedding_scaling': self.embedding_scaling,
            'input_length': self.input_length,
        })
        return config

# Load the model and tokenizer only once at module import time
model = None
tokenizer = None
label_encoder = None
max_length = None

def load_model_and_resources():
    global model, tokenizer, label_encoder, max_length
    
    # Load the model
    model = tf.keras.models.load_model('medical_model.h5', 
                                      custom_objects={'WeightedEmbedding': WeightedEmbedding})
    
    # Load the tokenizer
    import pickle
    with open('tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)
    
    # Load the label encoder
    with open('label_encoder.pickle', 'rb') as handle:
        label_encoder = pickle.load(handle)
    
    # Set the max_length (should match what was used in training)
    max_length = model.input_shape[1]

def process_symptoms(symptom_str):
    """
    Converts comma-separated symptoms into a list of standardized tokens.
    1) Lowercase
    2) Trim spaces
    3) Replace inner spaces with underscores
    """
    # Split on commas
    symptoms = symptom_str.lower().split(',')
    # Clean each symptom token
    symptoms = [s.strip().replace(' ', '_') for s in symptoms]
    return symptoms

def mapping(disease):
    """
    Maps a predicted disease to its description, precautions, medications,
    diet recommendations, and workout recommendations.
    """
    # Load the necessary dataframes
    description = pd.read_csv("description.csv")
    precautions = pd.read_csv("precautions_df.csv")
    medications = pd.read_csv("medications.csv")
    diets = pd.read_csv("diets.csv")
    workout = pd.read_csv("workout_df.csv")
    
    # Get description
    desc = description[description['Disease'] == disease]['Description']
    desc = " ".join([w for w in desc]) if not desc.empty else "No description available"
    
    # Get precautions
    pre = precautions[precautions['Disease'] == disease][['Precaution_1', 'Precaution_2', 'Precaution_3', 'Precaution_4']]
    pre = pre.values.flatten().tolist() if not pre.empty else []
    pre = [p for p in pre if isinstance(p, str) and p.strip()]  # Remove empty precautions
    
    # Get medications
    med = medications[medications['Disease'] == disease]['Medication']
    med = med.values.tolist() if not med.empty else []
    
    # Get diet recommendations
    diet = diets[diets['Disease'] == disease]['Diet']
    diet = diet.values.tolist() if not diet.empty else []
    
    # Get workout recommendations
    work = workout[workout['disease'] == disease]['workout']
    work = work.values.tolist() if not work.empty else []
    
    return desc, pre, med, diet, work

def predict_disease(symptoms_str):
    """
    Predicts a disease based on the input symptoms string.
    Returns a dictionary with the disease name, description, and recommendations.
    """
    global model, tokenizer, label_encoder, max_length
    
    # Load the model and resources if not already loaded
    if model is None:
        try:
            load_model_and_resources()
        except Exception as e:
            return {
                "error": f"Failed to load model or resources: {str(e)}"
            }
    
    try:
        # Process symptoms
        tokens = process_symptoms(symptoms_str)
        
        # Convert tokens to sequences
        sequences = tokenizer.texts_to_sequences(tokens)
        
        # Flatten the sequences
        flattened_seq = [idx for sublist in sequences for idx in sublist]
        
        # Pad the sequence
        padded_seq = pad_sequences([flattened_seq], maxlen=max_length, padding='post', dtype='int32')
        
        # Make prediction
        predictions = model.predict(padded_seq)
        
        # Get the top predicted disease
        top_index = np.argmax(predictions[0])
        predicted_disease = label_encoder.inverse_transform([top_index])[0]
        
        # Get disease information using mapping function
        description, precautions, medications, diet, workout = mapping(predicted_disease)
        
        # Create and return results dictionary
        result = {
            "disease": predicted_disease,
            "description": description,
            "precautions": precautions,
            "medications": medications,
            "diet": diet,
            "workout": workout
        }
        
        return result
    
    except Exception as e:
        return {
            "error": f"Prediction error: {str(e)}"
        }

# Uncomment to test the function
# if __name__ == "__main__":
#     test_symptoms = "fever, headache, fatigue"
#     result = predict_disease(test_symptoms)
#     print(result)