import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import streamlit as st
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


#   Görev 1: Data Analizi

# Veri setini yükle
@st.cache
def load_data():
    data = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data", header=None)
    return data

st.write('Veri Seti Yükleniyor...')
data = load_data()
data.columns = ['id', 'diagnosis', 'radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean',
                'compactness_mean', 'concavity_mean', 'concave points_mean', 'symmetry_mean', 'fractal_dimension_mean',
                'radius_se', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se', 'compactness_se', 'concavity_se',
                'concave points_se', 'symmetry_se', 'fractal_dimension_se', 'radius_worst', 'texture_worst',
                'perimeter_worst', 'area_worst', 'smoothness_worst', 'compactness_worst', 'concavity_worst',
                'concave points_worst', 'symmetry_worst', 'fractal_dimension_worst']

# İlk 10 satırı göster
st.subheader('İlk 10 Satır:')
st.write(data.head(10))

# Sütun isimlerini göster
st.subheader('Sütunlar:')
st.write(data.columns)


#   Görev 2: Data Ön İşlemleri

st.subheader('Veri Ön İşleme Tabi Tutuluyor...')
# Gereksiz sütunları temizle
# Analizde bir anlam ifade etmeği için 'id' sütununu kaldır
cleaned_data = data.drop(columns=['id'])

# Son 10 satırı göster
st.subheader('Son 10 Satır:')
st.write(cleaned_data.tail(10))

# 'diagnosis' sütunundaki değerleri M:1, B:0 yap
diagnosis_mapping = {'M': 1, 'B': 0}
cleaned_data["diagnosis"] = cleaned_data["diagnosis"].map(diagnosis_mapping)

# Korelasyon matrisini çizdir
st.subheader('Korelasyon Matrisi:')
corr_matrix = cleaned_data.corr()
sns.heatmap(corr_matrix, annot=True, cmap=sns.color_palette("rocket", as_cmap=True), fmt=".2f")
st.pyplot()

# Malignant ve benign olacak şekilde veriyi ayır ve çizdir
st.subheader('Malignant ve Benign:')
malignant_data = cleaned_data[cleaned_data["diagnosis"] == 1]
benign_data = cleaned_data[cleaned_data["diagnosis"] == 0]
sns.scatterplot(x=malignant_data["radius_mean"], y=malignant_data["texture_mean"], label='Malignant')
sns.scatterplot(x=benign_data["radius_mean"], y=benign_data["texture_mean"], label='Benign')
st.pyplot()

# Veriyi ayır ve ölçeklendir
X = cleaned_data.drop(columns=["diagnosis"]).values
y = cleaned_data["diagnosis"].values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, Y_train, Y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)


#   Görev 3: Model Gerçeklemesi

# Model seçimi
model_name = st.sidebar.selectbox("Model Seç:", ("KNN", "SVM", "Naive Bayes"))

st.write(f"Seçilen Model: {model_name}, Eğitiliyor...")
if model_name == "KNN":
    model = KNeighborsClassifier()
    params = {'n_neighbors': [5, 7, 9, 11, 15], 'weights': ['uniform', 'distance'], 'p': [2, 3, 4]}
elif model_name == "SVM":
    model = SVC()
    params = {'C': [10, 100, 500], 'gamma': [0.01, 0.001, 0.005], 'kernel': ['rbf', 'poly', 'sigmoid'], 'degree': [1, 2, 3], 'decision_function_shape': ['ovo', 'ovr']}
elif model_name == "Naive Bayes":
    model = GaussianNB()
    params = {}

# En iyi parametreleri bul
grid_search = GridSearchCV(model, params, cv=5)
grid_search.fit(X_train, Y_train)
best_params = grid_search.best_params_

# Optimum parametrelerle modeli eğit
st.write(f"En iyi parametreler: {best_params}")
model.set_params(**best_params)
model.fit(X_train, Y_train)


#   Görev 4: Model Analizi

# Model sonuçlarını göster
predictions = model.predict(X_test)
accuracy = accuracy_score(Y_test, predictions)
precision = precision_score(Y_test, predictions)
recall = recall_score(Y_test, predictions)
f1 = f1_score(Y_test, predictions)
conf_matrix = confusion_matrix(Y_test, predictions, labels=[1, 0])

st.subheader('Model Sonuçları:')
st.write(f"Accuracy: {accuracy}")
st.write(f"Precision: {precision}")
st.write(f"Recall: {recall}")
st.write(f"F1 Score: {f1}")
st.write('Confusion Matrix:')
st.write(conf_matrix)
