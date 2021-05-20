# %%
import pickle
loaded_model = pickle.load(open("model_pickle.dat", "rb"))

# %%
import pandas as pd
df = pd.read_csv('Incidents_Final.csv')
df = df.drop(columns = ['incident_date'])
#df['fraud_reported'].replace(to_replace='Y', value=1, inplace=True)
#df['fraud_reported'].replace(to_replace='N',  value=0, inplace=True)
bins = [-1, 3, 6, 9, 12, 17, 20, 24]  # Factorize according to the time period of the day.
names = ["past_midnight", "early_morning", "morning", 'fore-noon', 'afternoon', 'evening', 'night']
df['incident_period_of_day'] = pd.cut(df.incident_hour_of_the_day, bins, labels=names).astype(object)
df[['incident_hour_of_the_day', 'incident_period_of_day']].head(20)
dummies = pd.get_dummies(df[[
    'insured_sex', 
    'insured_education_level', 
    'incident_type', 
    'incident_severity',
    'incident_city',
    'auto_make', 
    'auto_model',
    'incident_period_of_day',
    'insured_hobbies'
]])
#dummies = dummies.join(df[[    "fraud_reported"]])
X = dummies
#y = dummies.iloc[:, -1]


# %%
y_pred_proba = loaded_model.predict_proba(X) #Predicting probability
y_pred = loaded_model.predict(X) #Predicting Frauds 

# %%
df1= pd.DataFrame(y_pred_proba, columns = ['Non_Fraud_Probability','Fraud_Probabilty'])
df["Fraud"]= pd.DataFrame(y_pred, columns = ['Fraud'])
df["Fraud_Probabilty"] = df1["Fraud_Probabilty"]*100
df.head(10)