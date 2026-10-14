# Option A: Predicting telecom customer churn

A simple assignment draft with real executed results. Start with Churn_Report.pdf.
The report is intentionally shortened to about 1,220 words, below the assignment's suggested 2,500 words, following the requested preference for brevity. This may affect marking.

## Easiest way to run the demo

[Open the notebook in Google Colab](https://colab.research.google.com/github/jb3bay-cmd/telco-churn-assignment/blob/main/churn_analysis.ipynb)

GitHub shows the notebook and stores files. Google Colab runs the code in your browser.

1. Open the Colab link above and sign in to Google if requested.
2. Select **Runtime > Run all**. If Colab asks whether you trust this notebook, review the code and confirm only if you are comfortable running it.
3. Wait until the cells finish. The first cell downloads the source dataset if needed. The analysis trains both models and recreates the saved model in your session.
4. Show section **2. Explore training data** and explain the charts.
5. Show section **4. Test results** and explain the model comparison.
6. Scroll to **6. Save and demonstrate**. Press the play button beside its code cell to run the prediction again.
7. Explain: “This customer has been with the company for 72 months and has a two year contract. The model estimates about a 4.6% chance of leaving, so it predicts they will stay. The actual outcome was that they stayed.”

Run all cells once before recording; then rerun only the demonstration cell while presenting. Running section 6 alone in a new session will fail because it needs variables created by earlier cells. The exact results may differ slightly with software versions. Colab uses its installed packages; the pinned requirements below describe the tested local environment. For reproducing the existing exported model exactly, use those versions locally. A Colab session is temporary, so download files you need to keep.

See [Demo_Notes.md](Demo_Notes.md) for a six minute speaking outline. No slides are required by the supplied brief.

## Included
- Churn_Report.pdf: four page report, with charts and references.
- Report_Editable.txt: editable report text.
- churn_analysis.ipynb: notebook with code and visible outputs.
- churn_model.joblib: fitted logistic regression pipeline.
- run_analysis.py: executable version of notebook code.
- data/telco.csv: downloaded source data.
- figures/: charts. results.csv and fairness.csv: measured results.
- Demo_Notes.md: timed outline for a 5 to 7 minute live or recorded demo.
- requirements.txt: tested package versions (Jupyter is needed only for interactive notebooks).

## Run
Use Python 3.11 or a compatible version. From this folder:

```bash
python -m pip install -r requirements.txt
python run_analysis.py
jupyter notebook churn_analysis.ipynb
```

The notebook already has outputs and does not need to be rerun just to view results. The script recreates the model and charts. A fixed random seed makes the split reproducible. Package versions are pinned for model compatibility. Only load trusted joblib files.

## Use the saved model
```python
import pandas as pd
import joblib
model = joblib.load('churn_model.joblib')
customer = pd.read_csv('data/telco.csv').drop(columns=['customerID', 'Churn']).iloc[[0]].copy()
customer['TotalCharges'] = pd.to_numeric(customer['TotalCharges'], errors='coerce')
customer['NewCustomer'] = (customer['tenure'] <= 12).astype(int)
print(model.predict_proba(customer)[0, 1])
```
The feature rule must match training. The saved pipeline contains imputation, encoding, scaling and the classifier. It was fitted on the training set; test customers were not used to train it.

## Results
Training/test split: 5,634 / 1,409. Model choice used training five fold CV F1.
Logistic regression CV F1: 0.593; decision tree: 0.539.
Test logistic regression: accuracy 80.2%, precision 65.8%, recall 52.9%, F1 58.7%.
Always stay baseline: 73.5% accuracy, zero churn recall.
The cutoff is 0.50. Churn is the positive class. No claims of causal effects or guaranteed savings.

## Dataset source documentation
Access date: 14 October 2026.
Assignment's suggested source: https://www.kaggle.com/datasets/blastchar/telco-customer-churn
Actual downloaded source: https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv
Repository: https://github.com/IBM/telco-customer-churn-on-icp4d
IBM sample description: https://www.ibm.com/docs/en/cognos-analytics/12.0.x?topic=samples-telco-customer-churn
This download has 7,043 rows and 21 columns. We did not compare it byte for byte with the Kaggle version. IBM describes its Telco churn sample as fictional. Use it for this educational exercise; do not represent it as current verified operational company data. Public access does not by itself establish unrestricted redistribution rights; check the source terms before publicly republishing data.

The bundled source CSV is unchanged. Preprocessing in code converts TotalCharges to numeric (11 blank values), drops customerID, maps Churn to 0/1 and adds NewCustomer. Median numeric imputation and most frequent categorical imputation are fitted inside training pipelines, followed by scaling and one hot encoding. No rows or valid outliers are removed. No duplicate IDs were present. Service absence categories are retained.

## Software and tools: references
Python Software Foundation. (n.d.). Python. https://www.python.org/
The pandas development team. (n.d.). pandas. https://pandas.pydata.org/
NumPy developers. (n.d.). NumPy. https://numpy.org/
Matplotlib development team. (n.d.). Matplotlib. https://matplotlib.org/
Scikit-learn developers. (n.d.). User guide. https://scikit-learn.org/stable/user_guide.html
Joblib developers. (n.d.). Joblib. https://joblib.readthedocs.io/
Project Jupyter. (n.d.). Jupyter. https://jupyter.org/
OpenAI. (2026). ChatGPT [AI assistant]. https://chatgpt.com/
Report generation uses ReportLab: https://www.reportlab.com/
All accessed or used 14 October 2026. Code and report were generated with ChatGPT assistance for this task, not copied from IBM's tutorial code. Group review and course appropriate disclosure are required; do not claim unaided authorship.

## Before submitting
1. Add four group members, student IDs and the submission date to the report.
2. Obtain and sign the official Ethics_Declaration.docx from the LMS. It was not supplied here. The assignment says it must be submitted before work begins; this package does not satisfy or claim that step.
3. Review the code and explain the model results yourselves. Check the course's AI assistance rules.
4. Submit this public repository link. Keep signed ethics forms and student IDs out of this public repository.
5. Record a 5 to 7 minute demo or deliver it live using Demo_Notes.md. No video has been recorded.
6. Check the due date with your instructor: the supplied brief states 8 October 2026.

## Suggested group roles (not a record of completed group work)
Member 1: dataset, cleaning and source documentation.
Member 2: charts and interpretation.
Member 3: models and evaluation.
Member 4: report and demo coordination.
Everyone: review findings, ethics and final submission.
