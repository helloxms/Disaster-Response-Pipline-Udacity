# Disaster Response Pipeline 

This is a test project for build data pipline
In the Project Workspace, we'll find a data set containing real messages that were sent during disaster events.   
We will create a machine learning pipeline to categorize these events so that you can send  
the messages to an appropriate disaster relief agency.

This project will include a web app where an emergency worker can input a new message and get classification  
results in several categories. The web app will also display visualizations of the data. 

## Project Components
There are three components in this project.
1. ETL Pipeline
This part of the data pipeline is the Extract, Transform, and Load process.

2. ML Pipeline
This portion is the machine learning portion, we will split the data into a training set and a test set.  
Then, we will create a machine learning pipeline that uses NLTK, as well as scikit-learn's Pipeline and  
GridSearchCV to output a final model that uses the message column to predict classifications for 36   
categories (multi-output classification). 
Finally, we will export the model to a pickle file. 

3. Flask Web App
In the last part, we'll display our results in a Flask web app. We have provided a workspace for users  
with starter files. We can upload our database file and pkl file with our model.

## Install & Run

```
python process_data.py disaster_messages.csv disaster_categories.csv DisasterResponse.db

python train_classifier.py ../data/DisasterResponse.db classifier.pkl

python run.py
Then visit the browser window for web page.
```

## Contributing
Udacity

## License

MIT © Udacity
