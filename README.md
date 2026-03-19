#### Project Environment setup 

How to create venv for this project to run in your local
Make sure you have these installed


### Prerequisites
`python == 3.9 version`

### creating virtual environment inside project repo
```
1. windows 
 python -m venv venv
 .\venv\Scripts\activate
 
2. Mac/Linux
 python -m venv venv
 source venv/bin/activate
```

#### Running the Project Pipeline
```angular2html
1. pip install -r requirements.txt

** Make sure the Mediapipe version is 0.10.9 to support `mediapipe solutions holistics` for the project.

2. Preprocess Data: python preprocess_dataset.py
3. Extract Skeletons: python extract_skeletons.py
(Extracts 225-dimensional 3D keypoints via MediaPipe Holistic)

4. Train Model: python train_model.py
(Trains the BiLSTM network using Focal Loss)

5. Test Model: python test_trained_model.py
(Evaluates test set and  generates metric plots)

```
### Result Overview
```angular2html
Evaluated on a 20% test test set across the 32 micro-gesture classes:

Accuracy(Overall): 60.87%

Mean F1-Score: 0.57

Key Insights: 
The model excels at capturing distinct kinematic movements (Classes 3 & 14 achieved > 0.90 F1). 
However, it struggles with highly subtle, localized finger variations that share identical starting poses (e.g., Classes 1, 10, and 19 scored 0.00).
```
