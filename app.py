import seaborn as sns
from sklearn.preprocessing import StandardScaler
import streamlit as st
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split, KFold


def App():
    #   Görev 0: Uygulama Tanıtımı
    # Uygulama başlığı
    st.title('Meme Kanseri Teşhisi')

    # Kullanılan kütüphane sürümleri:
    #   Streamlit: 1.16.0
    #   Numpy: 1.26.3
    #   Pandas: 2.1.4
    #   Seaborn: 0.12.2
    #   Scikit-learn: 1.2.2

    #   Görev 1: Data Analizi

    # Veri setini yükle
    @st.cache
    def load_data(file):
        data = pd.read_csv(file)
        return data

    uploaded_file = st.file_uploader("Upload CSV file", type=['csv'])
    st.write('Veri Seti Yükleniyor...')

    if uploaded_file is not None:
        data = load_data(uploaded_file)
        # data.columns = ['id', 'diagnosis', 'radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean',
        #                'compactness_mean', 'concavity_mean', 'concave points_mean', 'symmetry_mean', 'fractal_dimension_mean',
        #                'radius_se', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se', 'compactness_se', 'concavity_se',
        #                'concave points_se', 'symmetry_se', 'fractal_dimension_se', 'radius_worst', 'texture_worst',
        #                'perimeter_worst', 'area_worst', 'smoothness_worst', 'compactness_worst', 'concavity_worst',
        #                'concave points_worst', 'symmetry_worst', 'fractal_dimension_worst']
        # fractal_dimension_worst'dan sonra sütun varsa unnamed olarak isimlendir
        data.columns = data.columns[:32].tolist() + [f"Unnamed_{i}" for i in data.columns[32:]]

        # İlk 10 satırı göster
        st.subheader('İlk 10 Satır:')
        st.write(data.head(10))
        st.write(data.columns)

        # Sütun isimlerini göster
        st.subheader('Sütunlar:')
        st.write(data.columns)

        # Veri seti hakkında bilgi
        st.subheader('Veri Seti Hakkında Bilgi:')
        st.write(data.describe())

        #   Görev 2: Data Ön İşlemleri

        st.subheader('Veri Ön İşleme Tabi Tutuluyor...')
        # Gereksiz sütunları temizle
        # Analizde bir anlam ifade etmeği için 'id' sütununu kaldır
        cleaned_data = data.drop(columns=['id'])
        # Eğer varsa unnamed sütunlarını kaldır
        cleaned_data = cleaned_data.loc[:, ~cleaned_data.columns.str.contains('^Unnamed')]

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
        # Koralasyon matrisini inceledikten sonra aynı bilgiyi veren sütunları kaldır (perimeter, area, ve radius
        # sütunlarından sadece radius sütununu tut
        cleaned_data = cleaned_data.drop(
            columns=["perimeter_mean", "area_mean", "perimeter_se", "area_se", "perimeter_worst", "area_worst"])

        # Malignant ve benign olacak şekilde veriyi ayır ve çizdir
        st.subheader('Malignant ve Benign:')
        malignant_data = cleaned_data[cleaned_data["diagnosis"] == 1]
        benign_data = cleaned_data[cleaned_data["diagnosis"] == 0]
        sns.scatterplot(x=malignant_data["radius_mean"], y=malignant_data["texture_mean"], label='Malignant')
        sns.scatterplot(x=benign_data["radius_mean"], y=benign_data["texture_mean"], label='Benign')
        st.pyplot()
        # Teşhis ile aralarında korelasyonu olmayan sütunları kaldır
        cleaned_data = cleaned_data.drop(
            columns=["smoothness_mean", "compactness_mean", "symmetry_mean", "fractal_dimension_mean",
                     "texture_se", "smoothness_se", "compactness_se", "concavity_se", "concave points_se",
                     "symmetry_se", "compactness_worst", "fractal_dimension_worst"])

        # Veriyi ayır ve ölçeklendir
        num_folds = 5  # 5 katlı çapraz doğrulama, train/test split %80/%20 denk gelecek şekilde

        X = cleaned_data.drop(columns=["diagnosis"]).values
        y = cleaned_data["diagnosis"].values
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        X_train, X_test, Y_train, Y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
        # Train ve test veri setlerini ayırmak yerine çapraz doğrulama yap
        kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)

        #   Görev 3: Model Gerçeklemesi

        # Model seçimi
        model_name = st.sidebar.selectbox("Model Seç:", ("KNN", "SVM", "Gaussian Naive Bayes", "Bernoulli Naive Bayes"))

        st.write(f"Seçilen Model: {model_name}, Eğitiliyor...")
        if model_name == "KNN":
            model = KNeighborsClassifier()
            params = {'n_neighbors': [5, 7, 9, 11, 15], 'weights': ['uniform', 'distance'], 'p': [2, 3, 4]}
        elif model_name == "SVM":
            model = SVC()
            params = {'C': [10, 100, 500], 'gamma': [0.01, 0.001, 0.005], 'kernel': ['rbf', 'poly', 'sigmoid'],
                      'degree': [1, 2, 3], 'decision_function_shape': ['ovo', 'ovr']}
        elif model_name == "Gaussian Naive Bayes":
            model = GaussianNB()
            params = {}
        elif model_name == "Bernoulli Naive Bayes":
            model = BernoulliNB()
            params = {'alpha': [0.1, 0.5, 1.0], 'binarize': [0.0, 0.5, 1.0], 'fit_prior': [True, False]}

        # En iyi parametreleri bul
        grid_search = GridSearchCV(model, params, cv=5)
        grid_search.fit(X_train, Y_train)
        best_params = grid_search.best_params_

        # Optimum parametrelerle modeli eğit ve değerlendirme skorlarını kaydet
        scoring_metrics = ['accuracy', 'precision', 'recall',
                           'f1']  # Değerlendirme metrikleri: accuracy, precision, recall, f1

        st.write(f"En iyi parametreler: {best_params}")
        model.set_params(**best_params)
        # Her katlama için skorları ve confusion matrislerini depolamak için liste oluştur
        scores = {metric: [] for metric in scoring_metrics}
        confusion_matrices = []

        # Modeli eğit ve değerlendirme metriklerini hesapla
        for train_index, test_index in kf.split(X_scaled):
            X_train, X_test = X_scaled[train_index], X_scaled[test_index]
            y_train, y_test = y[train_index], y[test_index]

            # Modeli eğit
            model.fit(X_train, y_train)

            # Tahminleri yap
            y_pred = model.predict(X_test)

            # Confusion matrix hesapla
            cm = confusion_matrix(y_test, y_pred, labels=[1, 0])
            confusion_matrices.append(cm)

            # Değerlendirme metriklerini hesapla
            scores['accuracy'].append(accuracy_score(y_test, y_pred))
            scores['precision'].append(precision_score(y_test, y_pred))
            scores['recall'].append(recall_score(y_test, y_pred))
            scores['f1'].append(f1_score(y_test, y_pred))

        #   Görev 4: Model Analizi

        # Model sonuçlarını göster
        average_confusion_matrix = (sum(confusion_matrices) / len(confusion_matrices)).round().astype(int)
        average_scores = {metric: sum(values) / len(values) for metric, values in scores.items()}

        # Print the average confusion matrix and scores
        st.write("Average Scores:")
        for metric, score in average_scores.items():
            st.write(f"{metric.capitalize()}: {score}")
        st.write("Average Confusion Matrix:")
        st.write(average_confusion_matrix)
