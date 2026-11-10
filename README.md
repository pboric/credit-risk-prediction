# Credit Risk Prediction (Home Credit)

Predicting whether a loan applicant will default, using the Home Credit Default Risk data: a large, imbalanced
dataset (8 % defaults) spread over several linked tables. The project covers exploratory analysis, feature
engineering across the tables, tuned gradient-boosting models, a stacking ensemble, decision-threshold tuning and a
deployed prediction API.

## Results

ROC AUC on a held-out split of the training data (61,503 applicants):

| Model | ROC AUC |
|---|---:|
| LightGBM (tuned with Optuna) | 0.787 |
| XGBoost (tuned with Optuna) | 0.785 |
| Random Forest (tuned with Optuna) | 0.760 |
| Stacking ensemble of the three | 0.787 |

With only 8 % defaults, the default threshold of 0.5 almost never flags an applicant, so the threshold was tuned. At
a threshold of 0.2 the stacking model catches 39 % of defaulters with 28 % precision (ROC AUC 0.779 on the split
used for threshold tuning). This model and threshold are used in the API.

## Live API

The model is deployed with FastAPI at
[home-credit-group-defaulter-prediction.onrender.com](https://home-credit-group-defaulter-prediction.onrender.com)
(free hosting, so the first request can take up to a minute). `model_deployment/test_api.py` sends a test request.

## Repository

| File | Content |
|---|---|
| [`EDA.ipynb`](EDA.ipynb) | All tables: missing values, target imbalance, Phi-K and correlation analysis, categorical and continuous features |
| [`Feature_Engineering.ipynb`](Feature_Engineering.ipynb) | Cleaning and aggregating the bureau, previous application, POS cash, instalment and credit card tables; imputation; 453 features |
| [`Modelling.ipynb`](Modelling.ipynb) | Optuna tuning, model comparison, stacking, threshold tuning, saved models and a submission file |
| `utilities_eda.py` | Helper functions used in the EDA |
| `model_deployment/` | FastAPI service (`main.py`), Dockerfile, saved models and `test_api.py` |

## Data

[Home Credit Default Risk](https://www.kaggle.com/c/home-credit-default-risk/data) on Kaggle (about 2.5 GB, not
included). Put the CSV files in the folder the notebooks read from.

## Run it

```bash
pip install -r requirements.txt
# then run EDA.ipynb, Feature_Engineering.ipynb and Modelling.ipynb in order

# API, locally
cd model_deployment
pip install -r requirements.txt
uvicorn main:app --reload
```

## Limitations

- One engineered feature, the default rate among each applicant's 500 nearest neighbours, is computed on the full
  training data before the validation split. Each applicant's own label counts towards it with a weight of 1/500,
  so the validation scores may be slightly optimistic.
- At any threshold there is a clear trade-off: catching more defaulters means rejecting many good applicants.

## License

MIT
