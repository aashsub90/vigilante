# vigilante

### Instructions
#### 1. Get the data
  - Get the data from: 
  - Unzip 'data.zip' in root directory
  - Get stored graphs and utility matrix from: 
  - Unzip 'store.zip' in root directory
  - Create 'output' directory in root directory

#### 2. Run the code from the project root directory in the following order to build all the data objects and prediction models from scratch (OPTIONAL)
  - For data cleaning:<br/> 
    Usage:<br/> 
    `python data_cleaning.py`
    
  - For data preprocessing:<br/> 
    Usage:<br/> 
    `python data_preprocessing.py`
    
  - For training model:<br/> 
    Usage:<br/> 
    `python model_training.py`
    
#### 3. Run the code from the project test directory to test the model on sample data
  - Sample data: `sample.pkl` <br/>
  
  - Actual classification: `actual.pkl` <br/> 
  
  - Classification model: `randomForest_model.pkl` <br/>
    
  - Run the script:<br/> 
    Usage:<br/> 
    `python test_model.py`
  
#### 4. Run the notebooks from the project test directory in the following order to build all the data objects and prediction models from scratch (OPTIONAL)
  - For data exploration: `data_exploration.ipynb` <br/>     
  - For data cleaning: `data_cleaning.ipynb` <br/>     
  - For data preprocessing: `data_preprocessing.ipynb` <br/>
  - For data analysis: `data_analysis.ipynb` <br/>     
  - For training model: `model_training.ipynb` <br/>
  - For prediction: `data_prediction.iypnb` <br/>
