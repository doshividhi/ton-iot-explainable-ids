# %%
%pip install pandas
# %%
from pathlib import Path

import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
# %%
DATA_PATH = (
    Path.home()
    / "Downloads"
    / "cybersecurity"
    / "train_test_network.csv"
)

df = pd.read_csv(DATA_PATH)

print("Dataset shape:", df.shape)

# %%
DATA_PATH = (
    Path.home()
    / "Downloads"
    / "cybersecurity"
    / "train_test_network.csv"
)

df = pd.read_csv(DATA_PATH)

print("Dataset shape:", df.shape)

# %%
X = df.drop(columns=["label", "type"])

y_binary = df["label"]
y_multiclass = df["type"]

print("Feature shape:", X.shape)
print("Binary target shape:", y_binary.shape)
print("Multiclass target shape:", y_multiclass.shape)

# %%
categorical_columns = X.select_dtypes(
    include=["object", "category", "bool"]
).columns.tolist()

numerical_columns = X.select_dtypes(
    exclude=["object", "category", "bool"]
).columns.tolist()

print("Categorical columns:", len(categorical_columns))
print("Numerical columns:", len(numerical_columns))

# %%
(
    X_train,
    X_temp,
    y_binary_train,
    y_binary_temp,
    y_multiclass_train,
    y_multiclass_temp
) = train_test_split(
    X,
    y_binary,
    y_multiclass,
    test_size=0.30,
    stratify=y_binary,
    random_state=42
)

# %%
(
    X_validation,
    X_test,
    y_binary_validation,
    y_binary_test,
    y_multiclass_validation,
    y_multiclass_test
) = train_test_split(
    X_temp,
    y_binary_temp,
    y_multiclass_temp,
    test_size=0.50,
    stratify=y_binary_temp,
    random_state=42
)

print("Training rows:", len(X_train))
print("Validation rows:", len(X_validation))
print("Testing rows:", len(X_test))

# %%
numerical_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="median")
        ),
        (
            "scaler",
            StandardScaler()
        )
    ]
)

# %%
categorical_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(
                strategy="most_frequent"
            )
        ),
        (
            "encoder",
            OneHotEncoder(
                handle_unknown="ignore",
                min_frequency=10,
                sparse_output=True
            )
        )
    ]
)

# %%
preprocessor = ColumnTransformer(
    transformers=[
        (
            "numeric",
            numerical_pipeline,
            numerical_columns
        ),
        (
            "categorical",
            categorical_pipeline,
            categorical_columns
        )
    ]
)
print("Preprocessing workflow created.")
# %%


numerical_columns = X_train.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()

categorical_columns = X_train.select_dtypes(
    include=["object", "category", "bool"]
).columns.tolist()

print("Numerical columns:", numerical_columns)
print("Categorical columns:", categorical_columns)

# %%
X_train_encoded = preprocessor.fit_transform(
    X_train
)

X_validation_encoded = preprocessor.transform(
    X_validation
)

X_test_encoded = preprocessor.transform(
    X_test
)

print(
    "Encoded training shape:",
    X_train_encoded.shape
)

print(
    "Encoded validation shape:",
    X_validation_encoded.shape
)

print(
    "Encoded testing shape:",
    X_test_encoded.shape
)

# %%
MODEL_DIRECTORY = Path("models")

MODEL_DIRECTORY.mkdir(
    parents=True,
    exist_ok=True
)

joblib.dump(
    preprocessor,
    MODEL_DIRECTORY / "preprocessor.pkl"
)

print("Preprocessing workflow saved.")

# %%
print("Training binary class distribution:")

print(
    y_binary_train.value_counts(
        normalize=True
    )
)

print("\nTraining multiclass distribution:")

print(
    y_multiclass_train.value_counts()
)

# %%
print("Current working folder:")
print(Path.cwd())

print("\nPreprocessor saved at:")
print((MODEL_DIRECTORY / "preprocessor.pkl").resolve())

# %%
PROJECT_ROOT = Path.cwd().parent

MODEL_DIRECTORY = PROJECT_ROOT / "models"

MODEL_DIRECTORY.mkdir(
    parents=True,
    exist_ok=True
)

joblib.dump(
    preprocessor,
    MODEL_DIRECTORY / "preprocessor.pkl"
)

print(
    "Preprocessor saved at:",
    (MODEL_DIRECTORY / "preprocessor.pkl").resolve()
)
# %%
