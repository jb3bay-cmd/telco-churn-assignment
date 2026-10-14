# Generated from the notebook. Run from this project folder.
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn, joblib
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
Path('figures').mkdir(exist_ok=True)
df = pd.read_csv('data/telco.csv')
print('Rows and columns:', df.shape)
print(df.head(3).to_string(index=False))
print('Duplicate IDs:', df.customerID.duplicated().sum())
print('Missing before conversion:', int(df.isna().sum().sum()))

df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
print('Missing TotalCharges:', df.TotalCharges.isna().sum())
print('Tenure for missing bills:', df.loc[df.TotalCharges.isna(), 'tenure'].unique())
print(df[['tenure','MonthlyCharges','TotalCharges']].describe().round(2).to_string())
X = df.drop(columns=['customerID','Churn'])
y = df.Churn.map({'No':0, 'Yes':1})
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)
print('Training:',len(X_train),'Test:',len(X_test))
print('Overall churn:', round(y.mean()*100,2),'%')

eda = X_train.assign(Churn=y_train)
fig, axes = plt.subplots(1,3,figsize=(12,3.6))
for ax, col in zip(axes,['Contract','InternetService','SeniorCitizen']):
    rates = eda.groupby(col).Churn.mean().mul(100)
    rates.plot.bar(ax=ax,color='#227c9d')
    ax.set_title('Churn by '+col);ax.set_ylabel('Customers who churned (%)')
    ax.tick_params(axis='x',rotation=25)
plt.tight_layout();plt.savefig('figures/churn_groups.png',dpi=160,bbox_inches='tight');plt.show()
print('Contract churn (%):')
print(eda.groupby('Contract').Churn.mean().mul(100).round(1).to_string())
print('Average tenure and charges:')
print(eda.groupby('Churn')[['tenure','MonthlyCharges']].mean().round(2).to_string())
fig,axes=plt.subplots(1,2,figsize=(8,3.2))
for ax,col in zip(axes,['tenure','MonthlyCharges']):
    for label,name in [(0,'Stayed'),(1,'Left')]:
        ax.hist(eda.loc[eda.Churn==label,col],bins=20,alpha=.5,density=True,label=name)
    ax.set_xlabel(col);ax.set_ylabel('Density');ax.legend()
plt.tight_layout();plt.savefig('figures/distributions.png',dpi=160,bbox_inches='tight');plt.show()

# Apply the same simple feature rule to both sets.
X_train = X_train.copy(); X_test = X_test.copy()
for frame in [X_train,X_test]:
    frame['NewCustomer'] = (frame.tenure <= 12).astype(int)
numeric = X_train.select_dtypes(include='number').columns.tolist()
categorical = X_train.select_dtypes(exclude='number').columns.tolist()
def make_pipeline(model):
    prep = ColumnTransformer([
        ('num', Pipeline([('fill',SimpleImputer(strategy='median')),('scale',StandardScaler())]),numeric),
        ('cat', Pipeline([('fill',SimpleImputer(strategy='most_frequent')),('encode',OneHotEncoder(handle_unknown='ignore'))]),categorical)])
    return Pipeline([('prep',prep),('model',model)])
models = {
    'Logistic regression':make_pipeline(LogisticRegression(max_iter=2000,random_state=42)),
    'Decision tree':make_pipeline(DecisionTreeClassifier(max_depth=4,min_samples_leaf=30,random_state=42))}
cv = StratifiedKFold(n_splits=5,shuffle=True,random_state=42)
cv_scores = {name:cross_val_score(model,X_train,y_train,cv=cv,scoring='f1').mean() for name,model in models.items()}
print('Mean cross validation F1:', {k:round(v,3) for k,v in cv_scores.items()})
best_name = max(cv_scores,key=cv_scores.get)
print('Selected before test evaluation:',best_name)

rows=[]
for name,model in models.items():
    model.fit(X_train,y_train)
    pred = model.predict(X_test)
    rows.append([name,accuracy_score(y_test,pred),precision_score(y_test,pred,zero_division=0),recall_score(y_test,pred),f1_score(y_test,pred)])
base = np.zeros(len(y_test),dtype=int)
rows.append(['Always stay',accuracy_score(y_test,base),0,0,0])
results = pd.DataFrame(rows,columns=['Model','Accuracy','Precision','Recall','F1'])
print(results.round(3).to_string(index=False))
results.to_csv('results.csv',index=False)
best = models[best_name]
pred = best.predict(X_test)
cm = confusion_matrix(y_test,pred)
print('Confusion matrix (rows actual, columns predicted):'); print(cm)
ConfusionMatrixDisplay(cm,display_labels=['Stayed','Left']).plot(cmap='Blues',colorbar=False)
plt.title(best_name+' | test data');plt.tight_layout()
plt.savefig('figures/confusion.png',dpi=150,bbox_inches='tight');plt.show()

checks=[]
for group in [0,1]:
    mask = X_test.SeniorCitizen.eq(group)
    actual = y_test.loc[mask]; guesses = pred[mask.to_numpy()]
    tn,fp,fn,tp = confusion_matrix(actual,guesses,labels=[0,1]).ravel()
    checks.append([group,int(mask.sum()),round(guesses.mean(),3),round(tp/(tp+fn),3),round(fp/(fp+tn),3)])
fairness = pd.DataFrame(checks,columns=['SeniorCitizen','N','Flag rate','Recall','False positive rate'])
print(fairness.to_string(index=False))
fairness.to_csv('fairness.csv',index=False)

joblib.dump(best,'churn_model.joblib')
loaded = joblib.load('churn_model.joblib')
example = X_test.iloc[[0]]
risk = loaded.predict_proba(example)[0,1]
print(example[['tenure','MonthlyCharges','Contract','NewCustomer']].to_string(index=False))
print('Estimated churn probability:',round(float(risk),3))
print('Prediction:', 'Likely to leave' if risk >= 0.5 else 'Likely to stay')
print('Actual outcome:', 'Left' if y_test.iloc[0] else 'Stayed')
assert np.array_equal(loaded.predict(X_test),pred)
print('Saved model predictions verified.')