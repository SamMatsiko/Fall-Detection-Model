# Fall-Detection-Model
The dataset was obtained from Kaggle(https://www.kaggle.com/pitasr/falldata). As detailed on the website, this dataset was generated by wearable motion sensor units fit to the subjects’ body at six different positions. Each unit comprises of three tri-axial devices (accelerometer, gyroscope, and magnetometer/compass). Fourteen volunteers performed a standardized set of movements including 20 voluntary falls and 16 activities of daily living (ADLs), resulting in a large dataset with 16382 trials. The dataset comprises of 7 variables, namely; ACTIVITY,TIME, SL, EEG, BP, HR and CIRCULATION. Find details on each column.

ACTIVITY - activity classification
TIME - monitoring time
SL - sugar level
EEG - EEG monitoring rate
BP - Blood pressure
HR - Heart beat rate
CIRCLUATION - Blood circulation

The aim is to build a model that detects falls for people in the fall risk groups. With this dataset, i have built a model using 6 predictors to differentiate 6 human movements(captured under the target label variable, ACTIVITY) of Standing, Walking, Sitting, Falling, Cramps and Running that are represented by values of 0,1,2,3,4,5 repectively. 

The data has first been explored using the different technicques that have guided on the machine learning approaches to deploy. Three machine learning algorithms have been considered and the final testing coducted with the best performing algorithm, random forest. 
