# Medical Diagnosis Model Deployment

This repository contains a medical diagnosis model that predicts diseases based on symptoms.

## Files
- `medical_diagnosis_model.h5`: The trained TensorFlow model
- `medical_tokenizer.pickle`: The tokenizer for processing symptoms text
- `medical_label_encoder.pickle`: The label encoder for converting predictions to disease names
- `preprocessing_info.json`: Contains preprocessing parameters
- `severity_dict.pickle`: Dictionary of symptom severity weights
- `prediction_function.py`: Utility function for making predictions
- `app.py`: Flask API for serving predictions
- `requirements.txt`: Required Python packages

## Deployment Instructions

### Local Deployment
1. Install requirements:
   ```
   pip install -r requirements.txt
   ```

2. Run the Flask application:
   ```
   python app.py
   ```

3. Test the API:
   ```
   curl -X POST http://localhost:5000/predict \
   -H "Content-Type: application/json" \
   -d '{"symptoms": "Nausea, vomiting, chest_pain"}'
   ```

### Docker Deployment
1. Build Docker image:
   ```
   docker build -t medical-diagnosis-api .
   ```

2. Run Docker container:
   ```
   docker run -p 5000:5000 medical-diagnosis-api
   ```

### Cloud Deployment (AWS)
1. Package all files
2. Upload to AWS Elastic Beanstalk or set up on EC2
3. Configure security groups to allow traffic on port 5000
4. Set up a load balancer if needed for high availability

## Usage Example

```python
import requests

response = requests.post('http://localhost:5000/predict', 
                        json={'symptoms': 'Nausea, vomiting, chest_pain'})
                        
print(response.json())
```
