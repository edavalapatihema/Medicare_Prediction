import pandas as pd
import os
import time
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTENC
import joblib
from tqdm import tqdm

os.makedirs('models', exist_ok=True)
os.makedirs('data', exist_ok=True)

print("Loading dataset...")
try:
    df = pd.read_csv('data/healthcare_dataset_new_imbalanced.csv')
except FileNotFoundError:
    print("Error: Dataset not found at 'data/healthcare_dataset_new_imbalanced.csv'.")
    exit()

X = df.drop('disease', axis=1)
y = df['disease']

numerical_cols = ['age', 'bmi', 'systolic_bp', 'diastolic_bp', 'cholesterol', 'blood_glucose',
                  'heart_rate', 'weight_kg', 'height_cm', 'triglycerides', 'creatinine',
                  'hemoglobin', 'body_temp_c', 'wbc_count', 'platelet_count', 'sleep_hours']
categorical_cols = ['gender', 'smoking_status', 'region', 'insurance_type', 'family_history']

print("Splitting dataset...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("Imputing missing values before resampling...")

num_imputer = SimpleImputer(strategy="mean")
X_train_num = pd.DataFrame(
    num_imputer.fit_transform(X_train[numerical_cols]),
    columns=numerical_cols
)

cat_imputer = SimpleImputer(strategy="most_frequent")
X_train_cat = pd.DataFrame(
    cat_imputer.fit_transform(X_train[categorical_cols]),
    columns=categorical_cols
)

X_train_imputed = pd.concat([X_train_num, X_train_cat], axis=1)

print("Balancing dataset with SMOTENC...")
categorical_indices = [X_train_imputed.columns.get_loc(col) for col in categorical_cols]
smote_nc = SMOTENC(categorical_features=categorical_indices, random_state=42)

start_time = time.time()
for _ in tqdm(range(1), desc="Running SMOTENC"):
    X_resampled, y_resampled = smote_nc.fit_resample(X_train_imputed, y_train)
end_time = time.time()

print("Original dataset shape:", X_train.shape, y_train.shape)
print("Resampled dataset shape:", X_resampled.shape, y_resampled.shape)
print(f"SMOTENC completed in {end_time - start_time:.2f} seconds")

print("Setting up preprocessing...")
preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline(steps=[
            ('imputer', SimpleImputer(strategy="mean")),
            ('scaler', StandardScaler())
        ]), numerical_cols),
        ('cat', Pipeline(steps=[
            ('imputer', SimpleImputer(strategy="most_frequent")),
            ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ]), categorical_cols)
    ]
)

print("Building model pipeline...")
model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', VotingClassifier(estimators=[
        ('lr', LogisticRegression(max_iter=1000, class_weight={0:50, 1:1})),
        ('dt', DecisionTreeClassifier(random_state=42)),
        ('rf', RandomForestClassifier(random_state=42)),
        ('gbc', GradientBoostingClassifier(random_state=42)),
        ('svm', SVC(random_state=42, probability=True))
    ], voting='soft'))
])

print("Training model pipeline...")
start_time = time.time()
for _ in tqdm(range(1), desc="🔧 Fitting pipeline"):
    model_pipeline.fit(X_resampled, y_resampled)
end_time = time.time()
print(f"Training completed in {end_time - start_time:.2f} seconds")

print("Saving model and metadata...")
joblib.dump(model_pipeline, 'models/model_pipeline.joblib')
joblib.dump(df, 'data/dataset_for_eda.joblib')

category_mappings = {col: df[col].dropna().unique().tolist() for col in categorical_cols}
joblib.dump(category_mappings, 'data/category_mappings.joblib')

joblib.dump((X_test, y_test), 'data/test_data.joblib')

print("\n Training complete!")
print("Saved: models/model_pipeline.joblib")
print("       data/dataset_for_eda.joblib")
print("       data/category_mappings.joblib")
print("       data/test_data.joblib")