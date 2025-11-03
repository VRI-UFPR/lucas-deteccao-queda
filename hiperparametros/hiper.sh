#!/bin/bash

python3 models.py knn 1
python3 models.py knn 3
python3 models.py knn 5
python3 models.py knn 7
python3 models.py knn 9
python3 models.py knn 11
python3 models.py knn 15


python3 models.py rf 50 10
python3 models.py rf 50 20
python3 models.py rf 50 30
python3 models.py rf 50 None
python3 models.py rf 100 10
python3 models.py rf 100 20
python3 models.py rf 100 30
python3 models.py rf 100 None
python3 models.py rf 200 10
python3 models.py rf 200 20
python3 models.py rf 200 30
python3 models.py rf 200 None
python3 models.py rf 300 10
python3 models.py rf 300 20
python3 models.py rf 300 30
python3 models.py rf 300 None


python3 models.py svm linear 0.01
python3 models.py svm linear 0.1
python3 models.py svm linear 1.0
python3 models.py svm rbf 0.01 
python3 models.py svm rbf 0.1
python3 models.py svm rbf 1.0
python3 models.py svm poly 0.01 
python3 models.py svm poly 0.1
python3 models.py svm poly 1.0
python3 models.py svm sigmoid 0.01
python3 models.py svm sigmoid 0.1
python3 models.py svm sigmoid 1.0

python3 models.py mlp 5
python3 models.py mlp 10
python3 models.py mlp 20
python3 models.py mlp 30
python3 models.py mlp 40
python3 models.py mlp 50