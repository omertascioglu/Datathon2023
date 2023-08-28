import pandas as pd
from sklearn.preprocessing import LabelEncoder
from catboost import CatBoostClassifier

# Load and preprocess the data
train_set = pd.read_csv("train.csv")
test_set = pd.read_csv("test_x.csv")

cat_cols = [col for col in train_set.columns if str(train_set[col].dtypes) in ["category", "object", "bool"] and col != "Öbek İsmi"]
print(cat_cols)

train_set_encoded = pd.get_dummies(train_set, columns=cat_cols, drop_first=True)
test_set_encoded = pd.get_dummies(test_set, columns=cat_cols, drop_first=True)

train_set_without_target = train_set_encoded.drop(['Öbek İsmi'], axis=1)

# Perform label encoding on the target variable
label_encoder = LabelEncoder()
train_target_encoded = label_encoder.fit_transform(train_set_encoded["Öbek İsmi"])

# Train the CatBoost model
catboost_model = CatBoostClassifier()
catboost_model.fit(train_set_without_target, train_target_encoded)

# Predict the target values using the CatBoost model
tahmin_sonuclari_catboost = catboost_model.predict(test_set_encoded)

# Convert the numeric predictions back to original labels
predicted_labels_catboost = label_encoder.inverse_transform(tahmin_sonuclari_catboost)

# Save Predictions to CSV
veri = {
    'id': range(2340),
    'Öbek İsmi_CatBoost': predicted_labels_catboost
}
Tahmin_Dataframe = pd.DataFrame(veri)
Tahmin_Dataframe.set_index('id', inplace=True)

Tahmin_Dataframe.to_csv('submission.csv')
