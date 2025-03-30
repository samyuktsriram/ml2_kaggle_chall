import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from lightgbm import LGBMClassifier

# --- 1. Load Data ---
def load_data(train_path='german_credit_train.csv'):
    df = pd.read_csv(train_path)
    df['target'] = df['Risk'].map({'Risk': 1, 'No Risk': 0})
    print(f"Data loaded with shape: {df.shape}")
    print(f"Class distribution:\n{df['Risk'].value_counts(normalize=True)}")
    return df

# --- 2. Feature Engineering ---
def create_features(df, verbose=True):
    df_fe = df.copy()
    created_features = []
    df_fe['is_high_installment'] = (df_fe['InstallmentPercent'] >= 4).astype(int)
    df_fe['is_young'] = (df_fe['Age'] < 30).astype(int)
    df_fe['is_long_loan'] = (df_fe['LoanDuration'] > 24).astype(int)
    df_fe['is_large_loan'] = (df_fe['LoanAmount'] > 5000).astype(int)
    df_fe['is_long_residence'] = (df_fe['CurrentResidenceDuration'] > 2).astype(int)
    df_fe['no_checking'] = df_fe['CheckingStatus'].isin(['no checking account', '<0 DM', '0<=X<200 DM']).astype(int)
    df_fe['bad_credit'] = df_fe['CreditHistory'].isin(['critical account', 'delay in paying off']).astype(int)
    df_fe['long_employment'] = df_fe['EmploymentDuration'].isin(['>=7 years', '4<=X<7 years']).astype(int)
    high_risk_purposes = ['car (used)', 'business', 'radio/television', 'furniture/equipment']
    df_fe['risky_purpose'] = df_fe['LoanPurpose'].isin(high_risk_purposes).astype(int)
    df_fe['InstallmentRatio'] = df_fe['InstallmentPercent'] / 100 * df_fe['LoanAmount'] / df_fe['LoanDuration']
    df_fe['ShortTermLoan'] = (df_fe['LoanDuration'] <= 12).astype(int)
    df_fe['MediumTermLoan'] = ((df_fe['LoanDuration'] > 12) & (df_fe['LoanDuration'] <= 36)).astype(int)
    df_fe['LongTermLoan'] = (df_fe['LoanDuration'] > 36).astype(int)
    df_fe['YoungAge'] = (df_fe['Age'] < 30).astype(int)
    df_fe['MiddleAge'] = ((df_fe['Age'] >= 30) & (df_fe['Age'] < 50)).astype(int)
    df_fe['SeniorAge'] = (df_fe['Age'] >= 50).astype(int)
    df_fe['LoanAmountToAge'] = df_fe['LoanAmount'] / df_fe['Age']
    df_fe['CreditUtilization'] = df_fe['ExistingCreditsCount'] * df_fe['LoanAmount'] / (df_fe['Age'] - 18)
    df_fe['LoanBurden'] = df_fe['InstallmentPercent'] * (df_fe['Dependents'] + 1)
    df_fe['DebtToIncomeProxy'] = df_fe['InstallmentPercent'] * df_fe['ExistingCreditsCount']
    df_fe['FinancialStability'] = df_fe['long_employment'] * df_fe['is_long_residence']
    risk_factors = ['is_high_installment', 'is_young', 'is_long_loan', 'is_large_loan', 'no_checking', 'bad_credit', 'risky_purpose']
    df_fe['RiskFactorCount'] = df_fe[risk_factors].sum(axis=1)
    return df_fe

# --- 3. Prepare Data ---
def prepare_data_for_modeling(df_fe):
    categorical_cols = df_fe.select_dtypes(include=['object']).columns.tolist()
    categorical_cols.remove('Risk')
    df_encoded = df_fe.copy()
    for col in categorical_cols:
        le = LabelEncoder()
        df_encoded[col] = le.fit_transform(df_encoded[col])
    X = df_encoded.drop(['Risk', 'target'], axis=1)
    y = df_encoded['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    numerical_cols = X_train.select_dtypes(include=['float64', 'int64']).columns
    scaler = StandardScaler()
    X_train[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
    X_test[numerical_cols] = scaler.transform(X_test[numerical_cols])
    return X_train, X_test, y_train, y_test

# --- 4. Custom Objective Function ---
def custom_lgbm_objective(y_true, y_pred):
    y_prob = 1.0 / (1.0 + np.exp(-y_pred))
    real_prop = {'Risk': 0.02, 'No Risk': 0.98}
    train_prop = {'Risk': 1/3, 'No Risk': 2/3}
    pos_weight = real_prop['Risk'] / train_prop['Risk']
    neg_weight = real_prop['No Risk'] / train_prop['No Risk']
    weights = np.where(y_true == 1, pos_weight, neg_weight)
    grad = weights * (y_prob - y_true)
    hess = weights * y_prob * (1 - y_prob)
    return grad, hess

# --- 5. Train LightGBM Model ---
def train_lgbm_model(X_train, y_train, X_test, y_test):
    model = LGBMClassifier(
        objective='binary',  # Placeholder, overridden internally
        class_weight=None,
        n_estimators=1000,
        learning_rate=0.05,
        num_leaves=63,
        max_depth=7,
        random_state=42
    )
    print("Training LightGBM model with custom objective...")
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_test, y_test)],
        eval_metric='binary_logloss',
        callbacks=[
            lambda env: setattr(env.model, '_objective', lambda y_true, y_pred: custom_lgbm_objective(y_true, y_pred))
        ]
    )
    y_pred = model.predict(X_test)
    print("\nModel Performance:")
    print(classification_report(y_test, y_pred))
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['No Risk', 'Risk'], yticklabels=['No Risk', 'Risk'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig('confusion_matrix_lgbm.png', dpi=300)
    plt.close()
    return model

# --- 6. Main Function ---
def main():
    df = load_data()
    df_fe = create_features(df)
    X_train, X_test, y_train, y_test = prepare_data_for_modeling(df_fe)
    model = train_lgbm_model(X_train, y_train, X_test, y_test)
    return model

if __name__ == "__main__":
    main()

