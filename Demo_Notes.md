# Simple live demo outline: about 6 minutes
These are speaking notes, not a recorded demo. Rehearse with the notebook open. Explain in your own words and show the visible outputs.

## 0:00 to 0:45 | Problem
“We chose Option A. Churn means a customer leaves the company. We wanted to predict who may leave so that the company could offer support. We kept the project simple by comparing two basic models.”
Show the notebook title. Explain that the public IBM dataset is educational sample data.

## 0:45 to 1:35 | Data cleaning
Show the first two code outputs: 7,043 rows, 21 columns, no duplicate customer IDs and 11 blank total charges. Explain converting these to numbers and filling missing values inside training. Point out the removal of customerID. Explain that 20% of customers were held back for testing.

## 1:35 to 2:35 | Charts
Show contract and distribution charts. “Monthly contracts had 42.7% churn in the training set. Two year contracts had only 2.9%. Customers who left also had shorter average tenure and higher monthly charges.”
Explain that this is correlation, not proof that changing someone's contract prevents churn. Show the NewCustomer rule: tenure of 12 months or less.

## 2:35 to 3:35 | Models
Show the model definitions and cross validation output. “Logistic regression combines customer information to estimate a probability. A decision tree asks a sequence of questions. We limited the tree to keep it small. We selected the model using F1 on training cross validation, before checking the test set.”
Explain that preprocessing is part of each model pipeline.

## 3:35 to 4:35 | Results
Show the results table and confusion matrix. “Logistic regression had 80.2% accuracy. However, always guessing stay already gives 73.5%. Our model found 198 churners and missed 176. Of 301 alerts, 103 were customers who actually stayed.”
Explain precision and recall in everyday language. “We would use it as an initial signal, not as a final decision.”

## 4:35 to 5:15 | Working prediction
Run the saved model demonstration cell. Point out the customer's contract, tenure and monthly charge. Read the model's displayed probability and actual outcome. Explain the 50% cutoff. “This example shows that the saved model works. A single correct prediction does not prove the whole model is reliable.”

## 5:15 to 6:10 | Business action and limits
Show fairness output. Explain that senior customers have a higher false positive rate in this test. “We would start with voluntary support calls and compare retention against a control group. We would not automatically change prices or claim guaranteed savings. We need current company data and more fairness checks before deployment.”
Close with the team's actual contribution and AI disclosure as required by the course.
