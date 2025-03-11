import os
from flask import Flask, request, jsonify, send_file
from prediction_function import predict_disease

app = Flask(__name__)

@app.route('/')
def index():
    return send_file('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    
    if 'symptoms' not in data:
        return jsonify({'error': 'No symptoms provided'}), 400
    
    symptoms = data['symptoms']
    
    try:
        result = predict_disease(symptoms)
        
        if 'error' in result:
            return jsonify({'error': result['error']}), 500
            
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy'})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, host='0.0.0.0', port=port)
