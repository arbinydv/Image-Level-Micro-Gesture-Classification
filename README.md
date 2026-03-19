#### Project Environment setup 

How to create venv for this project to run in your local
Make sure you have these installed

python == 3.9 version

`creating virtual environment inside project repo`
```
1. windows 
 python -m venv venv
 .\venv\Scripts\activate
 
2. Mac/Linux
 python -m venv venv
 source venv/bin/activate
```

#### Running the Project
1. Install the project requirements and dependencies from the root of the repo
` pip install -r requirements.txt`

** Make sure the Mediapipe version is 0.10.9 to support `mediapipe solutions holistics` for the project.

2. Spin Off the project over CUDA/GPU if available else on CPU
3. [TRAINING + TESTING] There are XXXX files and how to start overall model building and training, follow the below steps

##### some data for report

data split upfront and then data preprocessing before extracting skeleton
train - 80% 
test - 20% 

The classes having less than 2 videos will be used for training purpose.
Result after preprocessing and data split 
  train : 32 classes | 81552 frames
  test  : 31 classes | 20528 frames


#### Architecture of the Model ( BiLSTM model)
```angular2html
Input: (Batch, 16, 225)        ← 16 frames, 225 keypoints each
         ↓
BiLSTM (128 hidden, 2 layers)  ← reads sequence forward + backward
         ↓
Concat final states → (256,)   ← captures full temporal context
         ↓
Linear(256→128) + BN + ReLU + Dropout
         ↓
Linear(128→32)                 ← 32 gesture classes
```
