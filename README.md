# Workflow CI

Repository ini berisi implementasi Continuous Integration (CI) Workflow menggunakan MLflow Project untuk melakukan re-training model clustering secara otomatis dengan GitHub Actions.

Pipeline ini mencakup setup environment, instalasi dependency, eksekusi training model, logging eksperimen MLflow, serta penyimpanan artifact model.

---

## Tujuan Proyek

- Mengotomatisasi proses training model Machine Learning
- Menerapkan MLflow sebagai alat tracking eksperimen
- Menyimpan artifact model secara terstruktur

---

## Dependencies
Environment dikelola menggunakan Conda dengan dependencies:

- Python 3.10
- MLflow 3.7.0
- Scikit-learn ≥ 1.4
- Pandas ≥ 2.2.3
- Numpy ≥ 1.26
- Matplotlib ≥ 3.8
- Seaborn ≥ 0.13.2
- Yellowbrick ≥ 1.5
- Joblib ≥ 1.3

---
