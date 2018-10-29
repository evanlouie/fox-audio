# AudioSet Inference Report

## Models

Note: LSTM AudioSet (Bal + Unbal) at target 0.01 loss is still training. Results to come soon.

| Model                    | Dataset                 | Steps | Loss  | Avg Hit | Avg PERR | MAP   | GAP   | Avg Loss  |
| ------------------------ | ----------------------- | ----- | ----- | ------- | -------- | ----- | ----- | --------- |
| LSTM                     | AudioSet (Bal)          | 4010  | ~0.01 | 0.481   | 0.319    | 0.157 | 0.273 | 18.990448 |
| LSTM (Adaptive Learning) | AudioSet + Fox Gunshots | 6130  | ~0.01 | 0.482   | 0.321    | 0.157 | 0.275 | 19.507206 |
| LSTM                     | AudioSet (Bal + Unbal)  | 4010  | ~4.00 | 0.527   | 0.341    | 0.236 | 0.337 | 9.632698  |
| LSTM                     | AudioSet (Bal + Unbal)  | 18020 | ~3.35 | 0.562   | 0.377    | 0.273 | 0.378 | 9.484769  |
| LSTM                     | AudioSet (Bal + Unbal)  | 56040 | ~0.06 | 0.529   | 0.341    | 0.189 | 0.299 | 17.330322 |
| LSTM                     | AudioSet (Bal + Unbal)  | 18020 | ~0.01 |         |          |       |       |           |


## Data Used

- AudioSet balanced data-set (22,176 total samples | 178 gunshots)
- AudioSet unbalanced data-set (2,042,985 total samples | 3,869 gunshots)
- Fox gunshots (257 clip)

## Runs

| Model                    | Dataset                        | Loss | Movie      | Correct Predictions | False Positive Gunshots |
| ------------------------ | ------------------------------ | ---- | ---------- | ------------------- | ----------------------- |
| LSTM                     | AudioSet (Bal)                 | 1.00 | Deadpool 1 | 9 / 14              | 36                      |
| LSTM (Adaptive Learning) | AudioSet (Bal) + Fox Gunshots* | 1.00 | Deadpool 1 | 7 / 14              | 25                      |
| LSTM                     | AudioSet (Bal)                 | 1.00 | Deadpool 2 | 5 / 25              | 29                      |
| LSTM (Adaptive Learning) | AudioSet (Bal) + Fox Gunshots* | 1.00 | Deadpool 2 | 5 / 25              | 27                      |
| LSTM                     | AudioSet (Bal)                 | 0.01 | Deadpool 1 | 0 / 14              | 11                      |
| LSTM (Adaptive Learning) | AudioSet (Bal) + Fox Gunshots* | 0.01 | Deadpool 1 | 1 / 14              | 9                       |
| LSTM                     | AudioSet (Bal)                 | 0.01 | Deadpool 2 | 0 / 25              | 14                      |
| LSTM (Adaptive Learning) | AudioSet (Bal) + Fox Gunshots* | 0.01 | Deadpool 2 | 2 / 25              | 11                      |
| LSTM                     | AudioSet (Bal + Unbal)         | 3.35 | Deadpool 1 | 3 / 14              | 19                      |
| LSTM                     | AudioSet (Bal + Unbal)         | 3.35 | Deadpool 2 | 9 / 25              | 24                      |
| LSTM                     | AudioSet (Bal + Unbal)         | 0.60 | Deadpool 1 | 1 / 14              | 9                       |
| LSTM                     | AudioSet (Bal + Unbal)         | 0.60 | Deadpool 2 | 3 / 25              | 14                      |
| LSTM                     | AudioSet (Bal + Unbal)         | 0.01 | Deadpool 1 | Coming soon         | Coming soon             |
| LSTM                     | AudioSet (Bal + Unbal)         | 0.01 | Deadpool 2 | Coming soon         | Coming soon             |

*Refer to bottom to see list of used audio samples

## Interpretation

Some interesting findings were made from this experiment especially in regards to potential 
over-training of the model. We see a large decrease in overall predictions when the model training 
continued to a loss of 1.00 to 0.01. This may be an excellent place for Fox to continue 
experimentation to find an optimal loss level. Fox's gunshot data did prove to overall beneficial to model 
performance, decrease false positives at a greater rate than decrease correct predictions. 

General Summary:

- Fox data generally improves model performance (correct predictions decreases at lower rate than false positives)
- Fox data has positive effect on false positives
- Model training from 1 -> 0.01 Loss caused drastic decrease in both true and false positive detections
- Fox gunshot samples are generally more "cluttered" than those in AudioSet (ie. Has more background noises such as music and speech). This may have negative effects on the model
- More experimentation is needed in loss levels between 1.00 - 0.01. 0.01 has proved to be overtrained.

## Next Steps

Only the balanced training set was utilized during training. This left a large amount of unused data unused for training.
The unbalanced training set was not used as to not make the model weights imbalanced, however it could prove useful to 
try and utilize the unbalanced training in addition to the balanced.

Transfer Learning is an avenue which should definitely be investigated as well. Utilizing a frozen LSTM model based on
AudioSet and retraining the top layers to only detect gunshots using only Fox data may prove much more effective than
the Adaptive Learning method used in this report.

## Fox Gunshot Data

A total of 257 10 second Fox movie audio samples were added during adaptive training.

- deadpool1_00-07-50.000.wav
- deadpool1_00-07-50.000.wav
- deadpool1_00-11-00.000.wav
- deadpool1_00-11-10.000.wav
- deadpool1_00-11-40.000.wav
- deadpool1_00-12-00.000.wav
- deadpool1_00-12-10.000.wav
- deadpool1_00-12-20.000.wav
- deadpool1_00-12-30.000.wav
- deadpool1_00-12-40.000.wav
- deadpool1_00-13-00.000.wav
- deadpool1_01-00-20.000.wav
- deadpool1_01-01-20.000.wav
- deadpool1_01-35-20.000.wav
- deadpool2_00-02-00.000.wav
- deadpool2_00-03-20.000.wav
- deadpool2_00-04-20.000.wav
- deadpool2_00-04-50.000.wav
- deadpool2_00-04-50.000.wav
- deadpool2_00-05-40.000.wav
- deadpool2_00-11-50.000.wav
- deadpool2_00-12-30.000.wav
- deadpool2_00-20-30.000.wav
- deadpool2_00-33-10.000.wav
- deadpool2_00-33-20.000.wav
- deadpool2_00-35-50.000.wav
- deadpool2_00-41-50.000.wav
- deadpool2_00-42-10.000.wav
- deadpool2_00-42-50.000.wav
- deadpool2_00-43-20.000.wav
- deadpool2_00-43-30.000.wav
- deadpool2_00-45-10.000.wav
- deadpool2_00-45-50.000.wav
- deadpool2_01-05-10.000.wav
- deadpool2_01-05-50.000.wav
- deadpool2_01-07-00.000.wav
- deadpool2_01-08-10.000.wav
- deadpool2_01-10-10.000.wav
- deadpool2_01-31-10.000.wav
- deadpool2_01-36-20.000.wav
- deadpool2_01-53-00.000.wav
- WarPOA_00-06-20.000.wav
- WarPOA_00-06-30.000.wav
- WarPOA_00-07-10.000.wav
- WarPOA_00-07-40.000.wav
- WarPOA_00-07-50.000.wav
- WarPOA_00-08-00.000.wav
- WarPOA_00-08-50.000.wav
- WarPOA_00-09-10.000.wav
- WarPOA_00-09-20.000.wav
- WarPOA_00-22-50.000.wav
- WarPOA_00-24-30.000.wav
- WarPOA_00-31-10.000.wav
- WarPOA_00-46-00.000.wav
- WarPOA_01-15-20.000.wav
- WarPOA_01-37-30.000.wav
- WarPOA_01-54-00.000.wav
- WarPOA_01-54-10.000.wav
- WarPOA_01-54-20.000.wav
- WarPOA_01-54-30.000.wav
- WarPOA_01-59-40.000.wav
- WarPOA_01-59-50.000.wav
- WarPOA_02-00-00.000.wav
- WarPOA_02-00-20.000.wav
- WarPOA_02-00-30.000.wav
- WarPOA_02-00-40.000.wav
- WarPOA_02-00-50.000.wav
- WarPOA_02-02-40.000.wav
- WarPOA_02-03-20.000.wav
- WarPOA_02-03-30.000.wav
- WarPOA_02-03-40.000.wav
- WarPOA_02-03-50.000.wav
- WarPOA_02-04-00.000.wav
- KingGC_00-03-50.000.wav
- KingGC_00-04-00.000.wav
- KingGC_00-04-30.000.wav
- KingGC_00-07-20.000.wav
- KingGC_00-07-30.000.wav
- KingGC_00-07-40.000.wav
- KingGC_00-07-50.000.wav
- KingGC_00-08-00.000.wav
- KingGC_00-08-10.000.wav
- KingGC_00-08-20.000.wav
- KingGC_00-08-30.000.wav
- KingGC_00-08-50.000.wav
- KingGC_00-21-10.000.wav
- KingGC_00-28-40.000.wav
- KingGC_00-28-50.000.wav
- KingGC_00-43-00.000.wav
- KingGC_01-09-50.000.wav
- KingGC_01-31-30.000.wav
- KingGC_01-31-40.000.wav
- KingGC_01-31-50.000.wav
- KingGC_01-32-00.000.wav
- KingGC_01-32-10.000.wav
- KingGC_01-32-20.000.wav
- KingGC_01-32-30.000.wav
- KingGC_01-32-40.000.wav
- KingGC_01-34-10.000.wav
- KingGC_01-34-20.000.wav
- KingGC_01-34-30.000.wav
- KingGC_01-34-40.000.wav
- KingGC_01-34-50.000.wav
- KingGC_01-36-20.000.wav
- KingGC_01-36-30.000.wav
- KingGC_01-46-00.000.wav
- KingGC_01-53-00.000.wav
- KingGC_01-53-10.000.wav
- KingGC_01-53-40.000.wav
- KingGC_01-54-20.000.wav
- KingGC_01-54-30.000.wav
- KingGC_01-54-40.000.wav
- KingGC_01-54-50.000.wav
- KingGC_01-55-00.000.wav
- KingGC_01-55-20.000.wav
- KingGC_01-55-30.000.wav
- KingGC_01-55-40.000.wav
- KingGC_01-55-50.000.wav
- KingGC_01-56-00.000.wav
- KingGC_01-56-10.000.wav
- KingGC_01-56-20.000.wav
- KingGC_01-56-30.000.wav
- KingGC_01-56-50.000.wav
- KingGC_01-57-00.000.wav
- KingGC_01-57-20.000.wav
- KingGC_01-57-30.000.wav
- KingGC_02-08-30.000.wav
- KingGC_02-08-40.000.wav
- KingGC_02-09-20.000.wav
- KingGC_02-09-30.000.wav
- KingGC_02-09-40.000.wav
- Logan_00-03-30.000.wav
- Logan_00-04-20.000.wav
- Logan_00-04-40.000.wav
- Logan_00-38-30.000.wav
- Logan_00-38-40.000.wav
- Logan_00-39-00.000.wav
- Logan_00-39-10.000.wav
- Logan_00-40-00.000.wav
- Logan_00-41-00.000.wav
- Logan_00-41-10.000.wav
- Logan_00-41-20.000.wav
- Logan_00-41-50.000.wav
- Logan_00-42-00.000.wav
- Logan_00-42-10.000.wav
- Logan_00-42-20.000.wav
- Logan_00-56-10.000.wav
- Logan_01-26-10.000.wav
- Logan_01-28-10.000.wav
- Logan_01-28-20.000.wav
- Logan_01-28-30.000.wav
- Logan_01-28-40.000.wav
- Logan_01-28-50.000.wav
- Logan_01-29-00.000.wav
- Logan_01-29-10.000.wav
- Logan_01-29-20.000.wav
- Logan_01-29-30.000.wav
- Logan_01-29-40.000.wav
- Logan_01-29-50.000.wav
- Logan_01-30-00.000.wav
- Logan_01-30-10.000.wav
- Logan_01-30-20.000.wav
- Logan_01-30-30.000.wav
- Logan_01-30-40.000.wav
- Logan_01-30-50.000.wav
- Logan_01-31-00.000.wav
- Logan_01-31-10.000.wav
- Logan_01-31-20.000.wav
- Logan_01-31-30.000.wav
- Logan_01-59-30.000.wav
- Logan_01-59-40.000.wav
- Logan_01-59-50.000.wav
- Logan_02-00-00.000.wav
- Logan_02-00-10.000.wav
- Logan_02-00-30.000.wav
- Logan_02-00-40.000.wav
- Logan_02-00-50.000.wav
- Logan_02-01-10.000.wav
- Logan_02-01-20.000.wav
- Logan_02-01-30.000.wav
- Logan_02-01-40.000.wav
- Logan_02-04-10.000.wav
- Logan_02-04-20.000.wav
- Logan_02-08-00.000.wav
- Predat4_00-56-10.000.wav
- Predat4_00-54-50.000.wav
- Predat4_00-53-30.000.wav
- Predat4_00-53-40.000.wav
- Predat4_00-53-50.000.wav
- Predat4_00-54-00.000.wav
- Predat4_00-54-10.000.wav
- Predat4_00-54-20.000.wav
- Predat4_00-54-30.000.wav
- Predat4_00-54-40.000.wav
- Predat4_00-35-00.000.wav
- Predat4_00-35-10.000.wav
- Predat4_00-32-50.000.wav
- Predat4_00-33-00.000.wav
- Predat4_00-32-00.000.wav
- Predat4_00-32-10.000.wav
- Predat4_00-32-20.000.wav
- Predat4_00-30-20.000.wav
- Predat4_00-30-30.000.wav
- Predat4_00-30-40.000.wav
- Predat4_01-36-20.000.wav
- Predat4_01-36-30.000.wav
- Predat4_01-35-00.000.wav
- Predat4_01-35-10.000.wav
- Predat4_01-34-30.000.wav
- Predat4_01-34-40.000.wav
- Predat4_01-34-10.000.wav
- Predat4_01-34-20.000.wav
- Predat4_01-33-00.000.wav
- Predat4_01-33-10.000.wav
- Predat4_01-33-20.000.wav
- Predat4_01-32-20.000.wav
- Predat4_01-30-30.000.wav
- Predat4_01-30-40.000.wav
- Predat4_01-29-20.000.wav
- Predat4_00-07-50.000.wav
- Predat4_00-08-00.000.wav
- Predat4_01-28-00.000.wav
- Predat4_01-28-10.000.wav
- Predat4_01-26-40.000.wav
- Predat4_01-26-50.000.wav
- Predat4_01-27-00.000.wav
- Predat4_01-27-10.000.wav
- Predat4_01-26-00.000.wav
- Predat4_01-26-10.000.wav
- Predat4_01-26-20.000.wav
- Predat4_01-25-50.000.wav
- Predat4_01-24-00.000.wav
- Predat4_01-24-10.000.wav
- Predat4_01-24-20.000.wav
- Predat4_01-24-30.000.wav
- Predat4_01-24-40.000.wav
- Predat4_01-19-00.000.wav
- Predat4_01-19-10.000.wav
- Predat4_01-19-20.000.wav
- Predat4_01-19-30.000.wav
- Predat4_01-19-40.000.wav
- Predat4_01-19-50.000.wav
- Predat4_01-18-20.000.wav
- Predat4_01-18-30.000.wav
- Predat4_01-18-40.000.wav
- Predat4_01-18-50.000.wav
- Predat4_01-19-00.000.wav
- Predat4_01-10-30.000.wav
- Predat4_01-10-40.000.wav
- Predat4_01-10-50.000.wav
- Predat4_01-11-20.000.wav
- Predat4_01-08-30.000.wav
- Predat4_01-08-40.000.wav
- Predat4_01-08-50.000.wav
- Predat4_00-05-00.000.wav
- Predat4_00-05-10.000.wav
- Predat4_00-05-20.000.wav
- Predat4_00-05-30.000.wav
