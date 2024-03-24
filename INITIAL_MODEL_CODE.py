import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder, label_binarize
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc, roc_auc_score, make_scorer, f1_score
from imblearn.over_sampling import SMOTE
import seaborn as sns
import mrmr
import optuna
from optuna.visualization import plot_contour
import plotly as pio
import logging
from mrmr import mrmr_classif
from scipy import interp
from itertools import cycle
from sklearn.model_selection import StratifiedKFold


# Defining parameters specific to models, to ensure Optuna searches for correct parameters, when model is changed
model_param_spaces = {
    'RandomForestClassifier': {
        "n_estimators": (100, 1000),
        "max_depth": (10, 100),
        "min_samples_split": (2, 20),
        "min_samples_leaf": (1, 10),
        "max_features": ["auto", "sqrt", "log2"]
    },
    "LogisticRegression": {
        "C": (0.01, 100.0),
        "solver": ["liblinear", "lbfgs", "newton-cg", "sag", "saga"],
    },
    "DecisionTreeClassifier": {
        "max_depth": (10, 100),
        "min_samples_split": (2, 20),
        "min_samples_leaf": (1, 10),
        "max_features": ["auto", "sqrt", "log2"]
    },
    "SVC": {
        "C": (0.01, 100.0),
        "kernel": ["linear", "poly", "rbf", "sigmoid"],
        "degree": (2, 5), 
        "gamma": ["scale", "auto"],
    }
}


def main(ml_model, use_optuna, use_feature_selection, remove_enrolled_class, remove_sem2_data, use_encoding, class_weight="balanced"):

    # Constants
    categorical_features = ["Marital status", "Displaced", "Educational special needs", "Debtor", "Tuition fees up to date", "Gender", "Scholarship holder", "International", 
                            "Application mode", "Daytime/evening attendance\t", "Previous qualification", "Nacionality", "Mother's qualification", "Father's qualification",
                            "Mother's occupation", "Father's occupation"]
    
    numeric_features = ["Application order", "Previous qualification (grade)", "Admission grade", "Age at enrollment", "GDP", "Inflation rate", "Curricular units 1st sem (credited)", "Curricular units 1st sem (enrolled)", "Curricular units 1st sem (evaluations)", 
                    "Curricular units 1st sem (approved)", "Curricular units 1st sem (without evaluations)", "Curricular units 1st sem (grade)", "Curricular units 2nd sem (credited)", "Curricular units 2nd sem (enrolled)", "Curricular units 2nd sem (evaluations)", 
                    "Curricular units 2nd sem (approved)", "Curricular units 2nd sem (grade)", "Curricular units 2nd sem (without evaluations)"]
    
    cat_columns_to_one_hot = ["Course"]

    # The main pipeline
    df = pd.read_csv("/Users/elaginivan/CODING/STUDENT_DATASET.csv", delimiter=";")
    target_colname = "Target"

    y_test, y_prediction, y_score = pipeline(df=df, target_colname=target_colname, categorical_features=categorical_features, numeric_features=numeric_features,
                                    cat_columns_to_one_hot=cat_columns_to_one_hot, ml_model=ml_model, use_optuna=use_optuna, use_feature_selection=use_feature_selection, 
                                    remove_enrolled_class=remove_enrolled_class, remove_sem2_data=remove_sem2_data, use_encoding=use_encoding, class_weight=class_weight) 

    visualize_results(y_test, y_prediction)
    
    print("Accuracy Score:", accuracy_score(y_test, y_prediction))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_prediction))
    print("Classification Report:\n", classification_report(y_test, y_prediction))
    
    unique_classes = np.unique(y_test)
    y_test_binarized = label_binarize(y_test, classes=unique_classes)
    n_classes = y_test_binarized.shape[1]
    
    # Plotting the ROC curve
    if n_classes == 2 or n_classes == 1:  # Binary classification scenario
        fpr, tpr, _ = roc_curve(y_test, y_score[:, 1])
        roc_auc = auc(fpr, tpr)
        plt.figure()
        plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:0.2f})')
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve")
        plt.legend(loc="lower right")
        plt.show()
        print(f"ROC AUC score: {roc_auc:.3f}")
        
    else:  # Multi-class scenario
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for i, class_label in enumerate(unique_classes):
            fpr[i], tpr[i], _ = roc_curve(y_test_binarized[:, i], y_score[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        # Plotting
        plt.figure()
        colors = cycle(["aqua", "darkorange", "cornflowerblue"])
        for i, color in zip(range(n_classes), colors):
            plt.plot(fpr[i], tpr[i], color=color, lw=2, label=f'ROC curve of class {i} (area = {roc_auc[i]:0.2f})')
        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("Multi-class ROC Curve")
        plt.legend(loc="lower right")
        plt.show()

        # Calculate and print multi-class ROC AUC scores
        roc_auc_ovr = roc_auc_score(y_test_binarized, y_score, multi_class="ovr", average="macro")
        print(f"Overall ROC AUC Score (One-vs-Rest): {roc_auc_ovr:.3f}")
        roc_auc_ovo = roc_auc_score(y_test_binarized, y_score, multi_class="ovo", average="macro")
        print(f"Overall ROC AUC Score (One-vs-One): {roc_auc_ovo:.3f}")


def preprocess(train, test, categorical_features, numeric_features, cat_columns_to_one_hot, use_encoding):

    if not use_encoding:
        return train, test

    # Processing numerical features
    for num_feature in numeric_features:
        scaler = StandardScaler()
        train[num_feature] = scaler.fit_transform(train[[num_feature]])
        test[num_feature] = scaler.transform(test[[num_feature]])

    # Processing categorical features not in one-hot encoding list
    for cat_feature in [cf for cf in categorical_features if cf not in cat_columns_to_one_hot]:
        le = LabelEncoder()
        # Fit on the combined set of train and test data to ensure consistency
        le.fit(pd.concat([train[cat_feature], test[cat_feature]], axis=0, ignore_index=True))
        train[cat_feature] = le.transform(train[cat_feature])
        test[cat_feature] = le.transform(test[cat_feature])

    # One-hot encoding specified categorical features
    for cat_feature in cat_columns_to_one_hot:
        ohe = OneHotEncoder(drop="first", handle_unknown="ignore")
    
        train_encoded = ohe.fit_transform(train[[cat_feature]]).toarray()
        test_encoded = ohe.transform(test[[cat_feature]]).toarray()
    
        # Generating feature names for the one-hot encoded columns
        feature_names = ohe.get_feature_names_out([cat_feature])
        
        # Updating the DataFrame with the new columns
        train = train.join(pd.DataFrame(train_encoded, index=train.index, columns=feature_names))
        test = test.join(pd.DataFrame(test_encoded, index=test.index, columns=feature_names))
    
        # Dropping the original categorical column as it's now encoded
        train.drop(columns=[cat_feature], inplace=True)
        test.drop(columns=[cat_feature], inplace=True)

    return train, test


def feature_selection(train_data, train_target, test_data, K):
    
    # Returns top K feature names selected using MRMR algorithm
    
    top_features = mrmr_classif(X=train_data, y=train_target, K=K)
    print(f"Top {K} features selected by MRMR:")
    for feature in top_features:
        print(feature)
    return train_data[top_features], test_data[top_features]


def optimize_function(train_data, train_target, ml_model_name, n_trials=100):
    
    # Fetching the parameter space for the model
    param_space = model_param_spaces[ml_model_name]
    
    # Objective function for Optuna optimization
    def objective(trial): 
        params = {}
        for param_name, param_range in param_space.items():
            # Parameter suggestions based on type
            if param_name == "C":
                params[param_name] = trial.suggest_loguniform(param_name, param_range[0], param_range[1])
            elif isinstance(param_range, list):
                params[param_name] = trial.suggest_categorical(param_name, param_range)
            elif param_name in ["max_depth", "min_samples_split", "min_samples_leaf", "n_estimators", "n_neighbors", "degree"]:
                if len(param_range) == 3:
                    params[param_name] = trial.suggest_int(param_name, param_range[0], param_range[1], step=param_range[2])
                else:
                    params[param_name] = trial.suggest_int(param_name, param_range[0], param_range[1])
            else:
                params[param_name] = trial.suggest_float(param_name, param_range[0], param_range[1])

        # Model class evaluation from string
        model_class = eval(ml_model_name)

        # Conditionally adding random_state only if supported by the model
        model_params = params.copy()  # Make a copy of the parameters
        if "random_state" in model_class().get_params():
            model_params["random_state"] = 42
        
        # Adding optimal parameters to the model
        model = model_class(**model_params)

        # Scoring and cross-validation setup
        scorer = make_scorer(f1_score, average="macro")
        cv_strategy = StratifiedKFold(n_splits=5)
        score = cross_val_score(model, train_data, train_target, scoring=scorer, cv=cv_strategy, n_jobs=-1).mean()
        return score

    # Optuna study and optimization
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)
    
    # Contout plot for visualization
    contplot = optuna.visualization.plot_contour(study)
    contplot.update_layout(font=dict(size=14), xaxis_title_font=dict(size=14), yaxis_title_font=dict(size=14), width=2000, height=1500)
    contplot.show()
    
    print("Best parameters:", study.best_params)
    return study.best_params


def pipeline(df, target_colname, categorical_features, numeric_features, cat_columns_to_one_hot, remove_enrolled_class, remove_sem2_data, ml_model, use_optuna, use_feature_selection, use_encoding, class_weight, n_trials=100):
    
    df = df[df['Course'] != 171]
    
    if remove_enrolled_class:
        df = df[df[target_colname] != "Enrolled"] # Removing 'Enrolled' class from the dataset
        
    sem2_columns = ["Curricular units 2nd sem (credited)", "Curricular units 2nd sem (enrolled)", "Curricular units 2nd sem (evaluations)", 
                    "Curricular units 2nd sem (approved)", "Curricular units 2nd sem (grade)", "Curricular units 2nd sem (without evaluations)"]

    if remove_sem2_data:
        df = df.drop(sem2_columns, axis=1)
        # Also removing these columns from the numeric_features list to avoid the KeyError
        numeric_features = [nf for nf in numeric_features if nf not in sem2_columns]
                
    label_encoder = LabelEncoder()
    df["Target_encoded"] = label_encoder.fit_transform(df[target_colname]) # 0 - dropout, 1 - enrolled, 2 - graduate
    y = df["Target_encoded"]
    X = df.drop(["Target", "Target_encoded"], axis=1)
    
    # Splitting the data into train and test sets prior to preprocessing to avoid data leakage
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Preprocessing the data separately for train and test sets
    X_train, X_test = preprocess(X_train, X_test, categorical_features, numeric_features, cat_columns_to_one_hot, use_encoding)
    
    # Oversampling the minority class
    smote = SMOTE(random_state=42)
    X_train, y_train = smote.fit_resample(X_train, y_train)

    # Feature selection using MRMR algorithm
    if use_feature_selection: 
        K = 10
        X_train, X_test = feature_selection(X_train, y_train, X_test, K)
    
    # Optuna hyperparameter optimization
    best_params = {} 
    if use_optuna:  
        best_params = optimize_function(X_train, y_train, ml_model.__name__, n_trials)
    else:
        best_params = {}
    
    model_params = best_params
    if hasattr(ml_model(), "random_state"):  # Check if the model supports "random_state"
        model_params["random_state"] = 42
    
    # Using the model without class_weight for models that do not support it
    if "class_weight" in ml_model().get_params():
        model = ml_model(class_weight=class_weight, **model_params)
    else:
        model = ml_model(**model_params)

    # Model training
    model.fit(X_train, y_train) 
    y_prediction = model.predict(X_test)
    y_score = model.predict_proba(X_test) if hasattr(model, "predict_proba") else None

    return y_test, y_prediction, y_score


# Visualization function
def visualize_results(y_test, y_prediction):
    
    cm = confusion_matrix(y_test, y_prediction) # Confusion matrix
    fig, ax = plt.subplots(figsize=(8, 6))  
    cax = ax.matshow(cm, cmap=plt.cm.Blues) 
    plt.title("Confusion Matrix", pad=20, fontsize=20)
    fig.colorbar(cax)
    
    plt.xlabel("Predicted", fontsize=14) 
    plt.ylabel("True", fontsize=14)   
    plt.xticks(np.arange(len(np.unique(y_test))), np.unique(y_test), rotation=90)
    plt.yticks(np.arange(len(np.unique(y_test))), np.unique(y_test))
    
    for i in range(cm.shape[0]): # Adding the numbers to the confusion matrix plot
        for j in range(cm.shape[1]): 
            ax.text(j, i, format(cm[i, j], "d"),
                    ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2. else "black") 
            
    plt.show()


# Calling the main function
if __name__ == "__main__":
    main(ml_model=RandomForestClassifier, use_optuna=False, use_feature_selection=True, remove_enrolled_class=True, remove_sem2_data=False, use_encoding=True, class_weight="balanced")
