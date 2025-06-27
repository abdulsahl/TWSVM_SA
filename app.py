import streamlit as st
import pandas as pd
import numpy as np
import joblib
from scipy.optimize import minimize 


class TwinSVM:

    def __init__(self, C1=1.0, C2=1.0, gamma=1.0):
        self.C1, self.C2, self.gamma = C1, C2, gamma
        self.v1, self.v2 = None, None
        self.support_vectors_A, self.support_vectors_B = None, None
        self.fit_successful = False

    def _rbf_kernel(self, X1, X2):
        if X1.ndim == 1: X1 = X1.reshape(1, -1)
        if X2.ndim == 1: X2 = X2.reshape(1, -1)
        dist_sq = np.sum(X1**2, axis=1).reshape(-1, 1) + np.sum(X2**2, axis=1) - 2 * np.dot(X1, X2.T)
        return np.exp(-self.gamma * dist_sq)

    def fit(self, X, y):

        pass

    def predict(self, X):

        if not self.fit_successful: return np.ones(X.shape[0]) * -1
        dist1 = np.abs(np.hstack([self._rbf_kernel(X, self.support_vectors_A), np.ones((X.shape[0], 1))]) @ self.v1)
        dist2 = np.abs(np.hstack([self._rbf_kernel(X, self.support_vectors_B), np.ones((X.shape[0], 1))]) @ self.v2)
        return np.where(dist1 < dist2, 1, -1)

@st.cache_resource
def load_model_and_scaler():
    try:
        model = joblib.load('twinsvm_model.pkl')
        scaler = joblib.load('scaler.pkl')
        return model, scaler
    except Exception as e:
        st.error(f"Gagal memuat model: {e}")
        return None, None


model, scaler = load_model_and_scaler()


st.set_page_config(page_title="Prediksi Kanker Payudara", layout="centered")
st.title("🔬 Aplikasi Prediksi Kanker Payudara")
st.markdown("Menggunakan model **TwinSVM (Optimized)** dengan Optimisasi parameter **Simulated Annealing** untuk klasifikasi tumor Jinak (Benign) atau Ganas (Malignant).")

if model is None or scaler is None:
    st.error("GAGAL MEMUAT MODEL. Pastikan file 'twinsvm_model.pkl' dan 'scaler.pkl' ada di folder yang sama.")
else:
    st.sidebar.header("Input Fitur Tumor")
    feature_names = [
        'radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean',
        'compactness_mean', 'concavity_mean', 'concave points_mean', 'symmetry_mean',
        'fractal_dimension_mean', 'radius_se', 'texture_se', 'perimeter_se', 'area_se',
        'smoothness_se', 'compactness_se', 'concavity_se', 'concave points_se',
        'symmetry_se', 'fractal_dimension_se', 'radius_worst', 'texture_worst',
        'perimeter_worst', 'area_worst', 'smoothness_worst', 'compactness_worst',
        'concavity_worst', 'concave points_worst', 'symmetry_worst', 'fractal_dimension_worst'
    ]
    
    input_dict = {}
    for feature in feature_names:
        input_dict[feature] = st.sidebar.number_input(f'{feature.replace("_", " ").title()}', value=0.0, format="%.4f")

    if st.sidebar.button("Prediksi Sekarang", type="primary"):
        input_df = pd.DataFrame([input_dict])
        input_scaled = scaler.transform(input_df)
        
        prediction = model.predict(input_scaled)
        
        st.subheader("Hasil Prediksi")
        if prediction[0] == -1:
            st.success("✔️ JINAK (Benign)")
            st.info("Tumor diprediksi bersifat jinak.")
        else:
            st.error("❌ GANAS (Malignant)")
            st.warning("Tumor diprediksi bersifat ganas. Disarankan untuk konsultasi medis.")