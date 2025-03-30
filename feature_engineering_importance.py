
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings('ignore')

# Set plotting style
plt.style.use('seaborn-whitegrid')
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (14, 10)
plt.rcParams['font.size'] = 12

def load_data(train_path='german_credit_train.csv'):
    """Load the credit risk data."""
    df = pd.read_csv(train_path)
    
    # Create a binary target variable (1 for 'Risk', 0 for 'No Risk')
    df['target'] = df['Risk'].map({'Risk': 1, 'No Risk': 0})
    
    print(f"Data loaded with shape: {df.shape}")
    print(f"Class distribution:\n{df['Risk'].value_counts(normalize=True)}")
    
    return df

def create_features(df, verbose=True):
    """
    Implement all feature engineering techniques for the credit risk dataset.
    
    Parameters:
    -----------
    df : DataFrame
        Input DataFrame with raw features
    verbose : bool
        Whether to print information about created features
        
    Returns:
    --------
    df_fe : DataFrame
        DataFrame with all engineered features
    """
    # Create a copy to avoid modifying the original
    df_fe = df.copy()
    
    # Track created features
    created_features = []
    
    # ----------------------
    # 1. Binary Risk Indicators
    # ----------------------
    
    # High Installment Percentage → risky if ≥ 4
    df_fe['is_high_installment'] = (df_fe['InstallmentPercent'] >= 4).astype(int)
    created_features.append('is_high_installment')
    
    # Young borrower → risky if < 30
    df_fe['is_young'] = (df_fe['Age'] < 30).astype(int)
    created_features.append('is_young')
    
    # Long Loan Duration → risky if > 24 months
    df_fe['is_long_loan'] = (df_fe['LoanDuration'] > 24).astype(int)
    created_features.append('is_long_loan')
    
    # Large Loan Amount → risky if > 5000
    df_fe['is_large_loan'] = (df_fe['LoanAmount'] > 5000).astype(int)
    created_features.append('is_large_loan')
    
    # Long residence duration → less risky if > 2 years
    df_fe['is_long_residence'] = (df_fe['CurrentResidenceDuration'] > 2).astype(int)
    created_features.append('is_long_residence')
    
    # No checking account or low balance → risky
    df_fe['no_checking'] = df_fe['CheckingStatus'].isin(['no checking account', '<0 DM', '0<=X<200 DM']).astype(int)
    created_features.append('no_checking')
    
    # Poor credit history → risky
    df_fe['bad_credit'] = df_fe['CreditHistory'].isin(['critical account', 'delay in paying off']).astype(int)
    created_features.append('bad_credit')
    
    # Long employment duration → less risky
    df_fe['long_employment'] = df_fe['EmploymentDuration'].isin(['>=7 years', '4<=X<7 years']).astype(int)
    created_features.append('long_employment')
    
    # Flag for high-risk loan purposes
    high_risk_purposes = ['car (used)', 'business', 'radio/television', 'furniture/equipment']
    df_fe['risky_purpose'] = df_fe['LoanPurpose'].isin(high_risk_purposes).astype(int)
    created_features.append('risky_purpose')
    
    # ----------------------
    # 2. Advanced Features
    # ----------------------
    
    # Installment Ratio - monthly payment burden
    df_fe['InstallmentRatio'] = df_fe['InstallmentPercent'] / 100 * df_fe['LoanAmount'] / df_fe['LoanDuration']
    created_features.append('InstallmentRatio')
    
    # Loan duration categories
    df_fe['ShortTermLoan'] = (df_fe['LoanDuration'] <= 12).astype(int)
    df_fe['MediumTermLoan'] = ((df_fe['LoanDuration'] > 12) & (df_fe['LoanDuration'] <= 36)).astype(int)
    df_fe['LongTermLoan'] = (df_fe['LoanDuration'] > 36).astype(int)
    created_features.extend(['ShortTermLoan', 'MediumTermLoan', 'LongTermLoan'])
    
    # Age categories
    df_fe['YoungAge'] = (df_fe['Age'] < 30).astype(int)
    df_fe['MiddleAge'] = ((df_fe['Age'] >= 30) & (df_fe['Age'] < 50)).astype(int)
    df_fe['SeniorAge'] = (df_fe['Age'] >= 50).astype(int)
    created_features.extend(['YoungAge', 'MiddleAge', 'SeniorAge'])
    
    # Loan amount to age ratio
    df_fe['LoanAmountToAge'] = df_fe['LoanAmount'] / df_fe['Age']
    created_features.append('LoanAmountToAge')
    
    # Credit utilization proxy
    df_fe['CreditUtilization'] = df_fe['ExistingCreditsCount'] * df_fe['LoanAmount'] / (df_fe['Age'] - 18)
    created_features.append('CreditUtilization')
    
    # Loan burden (installment relative to dependents)
    df_fe['LoanBurden'] = df_fe['InstallmentPercent'] * (df_fe['Dependents'] + 1)
    created_features.append('LoanBurden')
    
    # Debt-to-Income Proxy
    df_fe['DebtToIncomeProxy'] = df_fe['InstallmentPercent'] * df_fe['ExistingCreditsCount']
    created_features.append('DebtToIncomeProxy')
    
    # Financial Stability Indicator
    df_fe['FinancialStability'] = df_fe['long_employment'] * df_fe['is_long_residence']
    created_features.append('FinancialStability')
    
    # Risk factor combining multiple indicators
    risk_factors = ['is_high_installment', 'is_young', 'is_long_loan', 'is_large_loan', 
                   'no_checking', 'bad_credit', 'risky_purpose']
    df_fe['RiskFactorCount'] = df_fe[risk_factors].sum(axis=1)
    created_features.append('RiskFactorCount')
    
    if verbose:
        print(f"Created {len(created_features)} new features:")
        for feature in created_features:
            print(f"- {feature}")
    
    return df_fe

def prepare_data_for_modeling(df_fe):
    """
    Prepare data for modeling by encoding categorical variables
    and splitting into train and test sets.
    """
    # Get categorical and numerical columns
    categorical_cols = df_fe.select_dtypes(include=['object']).columns.tolist()
    categorical_cols.remove('Risk')  # Remove target
    
    # Create encoded versions of categorical features
    df_encoded = df_fe.copy()
    
    for col in categorical_cols:
        le = LabelEncoder()
        df_encoded[col] = le.fit_transform(df_encoded[col])
    
    # Features and target
    X = df_encoded.drop(['Risk', 'target'], axis=1)
    y = df_encoded['target']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale numerical features
    numerical_cols = X_train.select_dtypes(include=['float64', 'int64']).columns
    scaler = StandardScaler()
    X_train[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
    X_test[numerical_cols] = scaler.transform(X_test[numerical_cols])
    
    print(f"Training data shape: {X_train.shape}")
    print(f"Test data shape: {X_test.shape}")
    
    return X_train, X_test, y_train, y_test

def train_model_with_class_balance(X_train, y_train, X_test, y_test):
    """
    Train a Random Forest model with class balancing and evaluate it.
    """
    # Calculate class weights
    # Adjust weights to reflect real-world class distribution
    real_prop = {'Risk': 0.02, 'No Risk': 0.98}
    train_prop = {'Risk': 1/3, 'No Risk': 2/3}
    
    # Class weights to correct for sampling bias
    pos_weight = real_prop['Risk'] / train_prop['Risk']
    neg_weight = real_prop['No Risk'] / train_prop['No Risk']
    
    # Scale to ensure total weight equals total samples
    scale_factor = len(y_train) / (y_train.sum() * pos_weight + (len(y_train) - y_train.sum()) * neg_weight)
    pos_weight *= scale_factor
    neg_weight *= scale_factor
    
    # Create sample weights
    sample_weight = np.ones(len(y_train))
    sample_weight[y_train == 1] = pos_weight  # 'Risk' class
    sample_weight[y_train == 0] = neg_weight  # 'No Risk' class
    
    # Train Random Forest with sample weights
    model = RandomForestClassifier(
        n_estimators=500,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features='sqrt',
        bootstrap=True,
        random_state=42,
        n_jobs=-1
    )
    
    print("Training Random Forest model with class balancing...")
    model.fit(X_train, y_train, sample_weight=sample_weight)
    
    # Evaluate
    y_pred = model.predict(X_test)
    
    print("\nModel Performance:")
    print(classification_report(y_test, y_pred))
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
               xticklabels=['No Risk', 'Risk'],
               yticklabels=['No Risk', 'Risk'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=300)
    plt.close()
    
    return model

def plot_feature_importance(model, X_train, top_n=20, save_path='feature_importance.png'):
    """
    Plot feature importance from the trained model.
    
    Parameters:
    -----------
    model : RandomForestClassifier
        Trained Random Forest model
    X_train : DataFrame
        Training data with feature names
    top_n : int
        Number of top features to display
    """
    # Get feature importance
    importance = model.feature_importances_
    feature_names = X_train.columns
    
    # Create a DataFrame for easier sorting
    feat_importance = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importance
    }).sort_values(by='Importance', ascending=False)
    
    # Take top N features
    top_features = feat_importance.head(top_n).copy()
    
    # Color engineered vs original features differently
    engineered_features = [
        'is_high_installment', 'is_young', 'is_long_loan', 'is_large_loan', 
        'is_long_residence', 'no_checking', 'bad_credit', 'long_employment', 
        'risky_purpose', 'InstallmentRatio', 'ShortTermLoan', 'MediumTermLoan', 
        'LongTermLoan', 'YoungAge', 'MiddleAge', 'SeniorAge', 'LoanAmountToAge', 
        'CreditUtilization', 'LoanBurden', 'DebtToIncomeProxy', 'FinancialStability',
        'RiskFactorCount'
    ]
    
    top_features['Feature_Type'] = top_features['Feature'].apply(
        lambda x: 'Engineered Feature' if x in engineered_features else 'Original Feature'
    )
    
    # Create plot
    plt.figure(figsize=(12, 10))
    ax = sns.barplot(
        x='Importance', 
        y='Feature', 
        hue='Feature_Type',
        data=top_features,
        palette=['#1f77b4', '#ff7f0e']
    )
    
    # Add value labels to the bars
    for i, p in enumerate(ax.patches):
        width = p.get_width()
        plt.text(
            width + 0.002, 
            p.get_y() + p.get_height()/2, 
            f'{width:.4f}', 
            ha='left', 
            va='center',
            fontweight='bold'
        )
    
    plt.title('Random Forest Feature Importance: Original vs Engineered Features', fontsize=16)
    plt.xlabel('Importance (Mean Decrease in Impurity)', fontsize=14)
    plt.ylabel('Feature', fontsize=14)
    plt.tight_layout()
    
    # Save the figure
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Feature importance plot saved to {save_path}")
    
    # Return the feature importance DataFrame
    return feat_importance

def analyze_engineered_features(feat_importance):
    """Analyze the effectiveness of engineered features."""
    # Define engineered features
    engineered_features = [
        'is_high_installment', 'is_young', 'is_long_loan', 'is_large_loan', 
        'is_long_residence', 'no_checking', 'bad_credit', 'long_employment', 
        'risky_purpose', 'InstallmentRatio', 'ShortTermLoan', 'MediumTermLoan', 
        'LongTermLoan', 'YoungAge', 'MiddleAge', 'SeniorAge', 'LoanAmountToAge', 
        'CreditUtilization', 'LoanBurden', 'DebtToIncomeProxy', 'FinancialStability',
        'RiskFactorCount'
    ]
    
    # Calculate totals
    engineered_importance = feat_importance[feat_importance['Feature'].isin(engineered_features)]['Importance'].sum()
    original_importance = feat_importance[~feat_importance['Feature'].isin(engineered_features)]['Importance'].sum()
    total_importance = engineered_importance + original_importance
    
    # Display results
    print("\nFeature Importance Analysis:")
    print(f"Engineered Features: {engineered_importance:.4f} ({engineered_importance/total_importance:.1%})")
    print(f"Original Features: {original_importance:.4f} ({original_importance/total_importance:.1%})")
    
    # Top 5 engineered features
    top_engineered = feat_importance[feat_importance['Feature'].isin(engineered_features)].head(5)
    print("\nTop 5 Engineered Features:")
    for i, (_, row) in enumerate(top_engineered.iterrows(), 1):
        print(f"{i}. {row['Feature']}: {row['Importance']:.4f}")
        
    # Create pie chart for feature importance distribution
    plt.figure(figsize=(10, 6))
    plt.pie(
        [engineered_importance, original_importance],
        labels=['Engineered Features', 'Original Features'],
        autopct='%1.1f%%',
        startangle=90,
        colors=['#ff7f0e', '#1f77b4'],
        explode=(0.1, 0)
    )
    plt.title('Contribution to Model Prediction Power', fontsize=14)
    plt.tight_layout()
    plt.savefig('feature_contribution_pie.png', dpi=300)
    print("Feature contribution pie chart saved to feature_contribution_pie.png")

def main():
    """Main execution function."""
    # Load data
    print("Step 1: Loading data...")
    df = load_data()
    
    # Create features
    print("\nStep 2: Performing feature engineering...")
    df_fe = create_features(df)
    
    # Prepare data for modeling
    print("\nStep 3: Preparing data for modeling...")
    X_train, X_test, y_train, y_test = prepare_data_for_modeling(df_fe)
    
    # Train model
    print("\nStep 4: Training Random Forest model with class balancing...")
    model = train_model_with_class_balance(X_train, y_train, X_test, y_test)
    
    # Analyze feature importance
    print("\nStep 5: Analyzing feature importance...")
    feat_importance = plot_feature_importance(model, X_train, top_n=25)
    
    # Analyze engineered features
    analyze_engineered_features(feat_importance)
    
    print("\nFeature engineering and importance analysis completed!")

if __name__ == "__main__":
    main()
