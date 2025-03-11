import requests

response = requests.post('http://localhost:5000/predict', 
                        json={'symptoms': 'Nausea, vomiting, chest_pain'})
                        
print(response.json())