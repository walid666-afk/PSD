import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import altair as alt
from sklearn.preprocessing import MinMaxScaler
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import PCA
from streamlit_option_menu import option_menu
from imblearn.over_sampling import SMOTE
import pickle

st.set_page_config(page_title='TUGAS UAS PENDATA')
st.write("---")
st.markdown("<h1 style='text-align: center;'>UAS PROYEK SAINT DATA</h1>", unsafe_allow_html=True)
st.write("---")




selected = option_menu(
    menu_title=None,
    options=["Description", "Preprocessing", "Classification", "Implementation"],
    icons=["house","calculator", "table", "send"],
    menu_icon=None,
    default_index=0,
    orientation="horizontal",
)

if selected == "Description" :
    st.markdown("<h1 style='text-align: center;'>DESKRIPSI DATASET</h1>", unsafe_allow_html=True)
    st.write("###### Dataset yang digunakan adalah dataset penyakit DBD pada RSI GARAM KALIANGET Kab.SUMENEP, dapat dilihat pada tabel dibawah ini:")
    df = pd.read_csv('datadbd.csv')
    st.dataframe(df)
    st.write(" Dataset ini berisi informasi tentang individu yang didiagnosis dengan Demam Berdarah Dengue (DBD), suatu kondisi yang ditandai oleh infeksi virus dengue yang dapat menyebabkan penurunan jumlah trombosit dalam darah. Data mencakup informasi demografis seperti umur, suhu tubuh, dan jenis kelamin (dikodekan sebagai 0 untuk laki-laki dan 1 untuk perempuan), serta pengukuran klinis seperti tingkat hemoglobin, hematokrit, jumlah trombosit, dan jumlah leukosit. Variabel target dalam dataset ini adalah keberadaan atau ketiadaan DBD (dikodekan sebagai 0 untuk tidak terkena DBD dan 1 untuk terkena DBD). Dataset ini dapat digunakan untuk mengeksplorasi hubungan antara variabel ini dan kejadian DBD, serta untuk mengembangkan model prediktif guna mengidentifikasi individu yang berisiko mengembangkan kondisi ini. ")
    st.write("---")
    st.write("By Walid Rijal Awali")
    st.write("© Copyright 2023.")

if selected == "Preprocessing":
    st.markdown("<h1 style='text-align: center;'>TRANSFORMASI DATA</h1>", unsafe_allow_html=True)
    st.write(" Pada proses transformasi data disini, dilakukan pada kolom yang bertipe kategorikal, dimana pada dataset yang digunakan berada pada kolom 'Jenis Kelamin', dan pada kolom target yaitu kolom 'DB' dimana data pada kolom tersebut akan diubah ke dalam bentuk biner 1 dan 0, untuk proses dapat dilihat dibawah ini :")
    st.subheader("Data sebelum Transformasi :")
    df = pd.read_csv('datadbd.csv')
    df
    st.subheader("Data setelah Transformasi :")
    # Transformasi data pada kolom jenis kelamin
    # Assuming 'Gender' is the column name for gender
    gender_mapping = {'P': 0, 'L': 1}  # You can customize this based on your data
    df['Jenis Kelamin'] = df['Jenis Kelamin'].map(gender_mapping)

    # Transformasi data pada kolom DB
    # Assuming 'DB' is the column name for DB
    # You can apply any transformation based on your requirements
    # For example, you can use Label Encoding or create dummy variables
    # Here, I'll assume DB is binary (0 or 1)
    db_mapping = {'Negatif': 0, 'Positif': 1}  # Customize based on your data
    df['DB'] = df['DB'].map(db_mapping)
    df
    st.markdown("<h1 style='text-align: center;'>BALANCING DATA</h1>", unsafe_allow_html=True)
    st.write(" Pada proses selanjutnya data yang digunakan tidak seimbang sehingga perlu dilakukannya balancing data, dalam proses ini menggunakan metode SMOTE, dimana dalam proses ini adalah memayorkan data minor, dapat dilihat pada data dibawah :")
    st.subheader("Jumlah Data pada setiap Class Sebelum SMOTE :")
    st.write(df)
    # Menghitung jumlah data positif dan negatif
    jumlah_positif = df[df['DB'] == 1].shape[0]
    jumlah_negatif = df[df['DB'] == 0].shape[0]

    # Menampilkan hasil
    st.write("Jumlah Data Positif :", jumlah_positif)
    st.write("Jumlah Data Positif :", jumlah_negatif)
    # Memisahkan fitur (X) dan target (y)
    st.subheader("Jumlah Data pada setiap Class Sesudah SMOTE :")
    X = df.drop('DB', axis=1)
    y = df['DB']
    # Apply SMOTE for balancing the dataset
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)
    # Hasil oversampling dalam DataFrame baru
    oversampled_data = pd.DataFrame(X_resampled, columns=X.columns)
    oversampled_data['DB'] = y_resampled
    oversampled_data
    # Menghitung jumlah data positif dan negatif

    jumlah_positif1 = oversampled_data[oversampled_data['DB'] == 1].shape[0]
    jumlah_negatif1 = oversampled_data[oversampled_data['DB'] == 0].shape[0]
    # Menampilkan hasil
    st.write("Jumlah Data Positif :", jumlah_positif1)
    st.write("Jumlah Data Positif :", jumlah_negatif1)
    st.markdown("<h1 style='text-align: center;'>NORMALISASI DATA</h1>", unsafe_allow_html=True)
    st.subheader(" Rumus Normalisasi Data :")
    st.image('rumus_normalisasi.png', use_column_width=False, width=250)

    st.markdown("""
    Dimana :
    - X = data yang akan dinormalisasi atau data asli
    - min = nilai minimum semua data asli
    - max = nilai maksimum semua data asli
    """)
    # Mendefinisikan Varible X dan Y
    X = oversampled_data.drop(columns=['DB'])
    y = oversampled_data['DB'].values
    oversampled_data
    st.subheader("Pemisahan Kolom DB Sebagai Atribut Target")
    X
    df_min = X.min()
    df_max = X.max()

    # NORMALISASI NILAI X
    scaler = MinMaxScaler()
    # scaler.fit(features)
    # scaler.transform(features)
    scaled = scaler.fit_transform(X)
    features_names = X.columns.copy()
    # features_names.remove('label')
    scaled_features = pd.DataFrame(scaled, columns=features_names)

    st.subheader('Hasil Normalisasi Data')
    st.write(scaled_features)

    st.subheader('Target Label')
    dumies = pd.get_dummies(oversampled_data.DB).columns.values.tolist()
    dumies = np.array(dumies)

    labels = pd.DataFrame({
        '1': [dumies[0]],
        '2': [dumies[1]]
    })

    st.write(labels)
    st.write("---")
    st.write("By Walid Rijal Awali")
    st.write("© Copyright 2023.")

if selected == "Classification":
    # Nilai X training dan Nilai X testing
    df = pd.read_csv('Data_UJI.csv')
    X = df.drop(columns=['DB'])
    y = df['DB'].values


    training, test = train_test_split(X, test_size=0.2, random_state=42)
    training_label, test_label = train_test_split(y, test_size=0.2, random_state=42)  # Nilai Y training dan Nilai Y testing
    with st.form("Modeling"):
        st.markdown("<h1 style='text-align: center;'>CALSSIFICATION MODELS</h1>", unsafe_allow_html=True)
        st.write("Pilihlah model yang akan dilakukan pengecekkan akurasi:")
        naive = st.checkbox('Naive Bayes')
        k_nn = st.checkbox('K-Nearest Neighboor')
        destree = st.checkbox('Decission Tree')

        submitted = st.form_submit_button("Submit")

        # NB
        GaussianNB(priors=None)

        # Fitting Naive Bayes Classification to the Training set with linear kernel
        gaussian = GaussianNB()
        gaussian = gaussian.fit(training, training_label)

        # Predicting the Test set results
        y_pred = gaussian.predict(test)

        y_compare = np.vstack((test_label, y_pred)).T
        gaussian.predict_proba(test)
        gaussian_akurasi = round(100 * accuracy_score(test_label, y_pred))
        # akurasi = 10

        # Gaussian Naive Bayes
        # gaussian = GaussianNB()
        # gaussian = gaussian.fit(training, training_label)

        # probas = gaussian.predict_proba(test)
        # probas = probas[:,1]
        # probas = probas.round()

        # gaussian_akurasi = round(100 * accuracy_score(test_label,probas))

        # KNN
        K = 5
        knn = KNeighborsClassifier(n_neighbors=K)
        knn.fit(training, training_label)
        knn_predict = knn.predict(test)

        knn_akurasi = round(100 * accuracy_score(test_label, knn_predict))

        # Decission Tree
        dt = DecisionTreeClassifier()
        dt.fit(training, training_label)
        # prediction
        dt_pred = dt.predict(test)
        # Accuracy
        dt_akurasi = round(100 * accuracy_score(test_label, dt_pred))

        if submitted:
            if naive:
                st.write('Model Naive Bayes accuracy score: {0:0.2f}'. format(
                    gaussian_akurasi))
            if k_nn:
                st.write(
                    "Model KNN accuracy score : {0:0.2f}" . format(knn_akurasi))
            if destree:
                st.write(
                    "Model Decision Tree accuracy score : {0:0.2f}" . format(dt_akurasi))

        grafik = st.form_submit_button("Grafik akurasi semua model")
        if grafik:
            data = pd.DataFrame({
                'Akurasi': [gaussian_akurasi, knn_akurasi, dt_akurasi],
                'Model': ['Naive Bayes', 'K-NN', 'Decission Tree'],
            })

            chart = (
                alt.Chart(data)
                .mark_bar()
                .encode(
                    alt.X("Akurasi"),
                    alt.Y("Model"),
                    alt.Color("Akurasi"),
                    alt.Tooltip(["Akurasi", "Model"]),
                )
                .interactive()
            )
            st.altair_chart(chart, use_container_width=True)
    st.write("---")
    st.write("By Walid Rijal Awali")
    st.write("© Copyright 2023.")

if selected=="Implementation":

    # Function to load models using pickle
    def load_models():
        with open('nbmodel.pkl', 'rb') as nb_model_file:
            gaussian = pickle.load(nb_model_file)

        with open('dtmodel.pkl', 'rb') as knn_model_file:
            knn = pickle.load(knn_model_file)

        with open('knnmodel.pkl', 'rb') as dt_model_file:
            dt = pickle.load(dt_model_file)

        with open('scaler.pkl', 'rb') as normalization_file:
            normalization = pickle.load(normalization_file)

        return gaussian, knn, dt, normalization

    # Load models
    gaussian, knn, dt, normalization = load_models()

    # Streamlit code
    with st.form("my_form"):
        st.markdown("<h1 style='text-align: center;'>IMPLEMENTASI</h1>", unsafe_allow_html=True)
        age = st.number_input("Umur (Tahun)", min_value=0, max_value=100, value=0)
        temperature = st.number_input("Suhu Tubuh (°C)", min_value=0.0, max_value=100.0, value=0.0)
        gender = st.selectbox("Jenis Kelamin", ["Laki-laki", "Perempuan"])
        hemoglobin = st.number_input("Hemoglobin (g/dL)", min_value=0.0, max_value=100.0, value=0.0)
        hematocrit = st.number_input("Hematokrit (%)", min_value=0.0, max_value=100.0, value=0.0)
        platelet_count = st.number_input("Jumlah Trombosit (ribu/uL)", min_value=0.0, max_value=300.0, value=0.0)
        white_blood_cell_count = st.number_input("Jumlah Leukosit (ribu/uL)", min_value=0.0, max_value=100.0, value=0.0)
        gender_numeric = 0 if gender == "Perempuan" else 1
        model = st.selectbox('Pilihlah model yang akan anda gunakan untuk melakukan prediksi dibawah ini:',
                            ('Naive Bayes', 'K-NN', 'Decision Tree'))

        prediksi = st.form_submit_button("Submit")

        if prediksi:
            inputs = np.array([
                age,
                temperature,
                gender_numeric,
                hemoglobin,
                hematocrit,
                platelet_count,
                white_blood_cell_count
            ])

            # Normalize the inputs
            inputs = normalization.transform([inputs])

            if model == 'Naive Bayes':
                mod = gaussian
            elif model == 'K-NN':
                mod = knn
            elif model == 'Decision Tree':
                mod = dt

            input_pred = mod.predict(inputs)

            
            st.subheader('Hasil Prediksi')
            st.write('Menggunakan Pemodelan :', model)
            st.write(input_pred)
            ada = 1
            tidak_ada = 0
            if input_pred == ada:
                st.write('Berdasarkan hasil Prediksi Menggunakan Permodelan ',
                        model, 'Pasien di Diagnosis penyakit DBD')
            else:
                st.write('Berdasarkan hasil Prediksi Menggunakan Permodelan ',
                        model, 'Pasien Tidak di Diagnosis penyakit DBD')

    st.write("---")
    st.write("By Walid Rijal Awali")
    st.write("© Copyright 2023.")
