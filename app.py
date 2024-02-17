import seaborn as sns
from sklearn.model_selection import train_test_split
import streamlit as st
import pandas as pd


#   Görev 1: Data Analizi

# Veri setini yükle
@st.cache
def load_data():
    data = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data", header=None)
    return data

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

# Veriyi ayır
X = cleaned_data.drop(columns=[1])
y = cleaned_data["diagnosis"]
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2, random_state=42)

