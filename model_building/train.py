
from sklearn.model_selection import train_test_split

train_df, test_df = train_test_split(
    df,
    test_size=0.2,
    random_state=42,
    stratify=df["ProdTaken"]
)

train_df.to_csv(
    "tourism_project/data/train.csv",
    index=False
)

test_df.to_csv(
    "tourism_project/data/test.csv",
    index=False
)

Dataset.from_pandas(
    train_df
).push_to_hub(
    "mangalachanda99/tourism-train"
)

Dataset.from_pandas(
    test_df
).push_to_hub(
    "mangalachanda99/tourism-test"
)

!pip install mlflow xgboost

!pip install mlflow xgboost joblib

from datasets import load_dataset

train_df = load_dataset(
    "mangalachanda99/tourism-train"
)["train"].to_pandas()

test_df = load_dataset(
    "mangalachanda99/tourism-test"
)["train"].to_pandas()

print(train_df.shape)
print(test_df.shape)

X_train = train_df.drop("ProdTaken", axis=1)
y_train = train_df["ProdTaken"]

X_test = test_df.drop("ProdTaken", axis=1)
y_test = test_df["ProdTaken"]

from sklearn.preprocessing import LabelEncoder

categorical_cols = X_train.select_dtypes(
    include="object"
).columns

encoders = {}

for col in categorical_cols:

    le = LabelEncoder()

    X_train[col] = le.fit_transform(X_train[col])

    X_test[col] = le.transform(X_test[col])

    encoders[col] = le

    import mlflow

mlflow.set_experiment(
    "Tourism_Package_Prediction"
)

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV

dt_params = {
    "max_depth":[3,5,10,None],
    "min_samples_split":[2,5,10]
}

dt_grid = GridSearchCV(
    DecisionTreeClassifier(),
    dt_params,
    cv=5,
    scoring="f1"
)

dt_grid.fit(X_train,y_train)

with mlflow.start_run(run_name="DecisionTree"):

    mlflow.log_params(
        dt_grid.best_params_
    )

    mlflow.log_metric(
        "best_cv_score",
        dt_grid.best_score_
    )
    
from sklearn.ensemble import RandomForestClassifier

rf_params = {

    "n_estimators":[100,200],

    "max_depth":[5,10,None]
}

rf_grid = GridSearchCV(

    RandomForestClassifier(),

    rf_params,

    cv=5,

    scoring="f1"
)

rf_grid.fit(X_train,y_train)

with mlflow.start_run(run_name="RandomForest"):

    mlflow.log_params(
        rf_grid.best_params_
    )

    mlflow.log_metric(
        "best_cv_score",
        rf_grid.best_score_
    )

    from xgboost import XGBClassifier

xgb_params = {

    "n_estimators":[100,200],

    "max_depth":[3,5],

    "learning_rate":[0.01,0.1]
}

xgb_grid = GridSearchCV(

    XGBClassifier(),

    xgb_params,

    cv=5,

    scoring="f1"
)

xgb_grid.fit(X_train,y_train)

with mlflow.start_run(run_name="XGBoost"):

    mlflow.log_params(
        xgb_grid.best_params_
    )

    mlflow.log_metric(
        "best_cv_score",
        xgb_grid.best_score_
    )

    from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

def evaluate_model(model):

    pred = model.predict(X_test)

    prob = model.predict_proba(X_test)[:,1]

    return {

        "accuracy":
        accuracy_score(y_test,pred),

        "precision":
        precision_score(y_test,pred),

        "recall":
        recall_score(y_test,pred),

        "f1":
        f1_score(y_test,pred),

        "roc_auc":
        roc_auc_score(y_test,prob)
    }

    results = {

    "Decision Tree":
    evaluate_model(
        dt_grid.best_estimator_
    ),

    "Random Forest":
    evaluate_model(
        rf_grid.best_estimator_
    ),

    "XGBoost":
    evaluate_model(
        xgb_grid.best_estimator_
    )
}

best_model = xgb_grid.best_estimator_

import joblib

joblib.dump(
    best_model,
    "best_model.pkl"
)

from huggingface_hub import HfApi

api = HfApi()

api.upload_file(

    path_or_fileobj=
    "best_model.pkl",

    path_in_repo=
    "best_model.pkl",

    repo_id=
    "mangalachanda99/tourism-model",

    repo_type="model"
)
