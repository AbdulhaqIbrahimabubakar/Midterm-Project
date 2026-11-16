import requests

url = 'http://localhost:9696/predict'

sample_student = {
    'school': 'gp', 'sex': 'f', 'age': 17, 'address': 'u', 'famsize': 'gt3',
    'pstatus': 't', 'medu': 4, 'fedu': 4, 'mjob': 'teacher', 'fjob': 'services',
    'reason': 'course', 'guardian': 'mother', 'traveltime': 1, 'studytime': 3,
    'failures': 0, 'schoolsup': 'no', 'famsup': 'yes', 'paid': 'no',
    'activities': 'yes', 'nursery': 'yes', 'higher': 'yes', 'internet': 'yes',
    'romantic': 'no', 'famrel': 4, 'freetime': 3, 'goout': 2, 'dalc': 1,
    'walc': 1, 'health': 5, 'absences': 2, 'g1': 14, 'g2': 15
}

response = requests.post(url, json=sample_student).json()
print("Prediction response:", response)
# test_predict.py (Test Client)