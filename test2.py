import streamlit as st
import pandas as pd
import numpy as np
import scikitplot as skplt
import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import classification_report, accuracy_score

# User-Interface
st.set_page_config(layout="wide",
                   page_title="Ensemble+IDS Study",
                   page_icon="🤖")
st.write('<style>div.block-container{padding-top:2rem;}</style>', unsafe_allow_html=True)

st.title("Comparative analysis of Voting Classifiers over IDS Datasets")
st.write("### A joint study of Ensemble Models and IDS.")
tab1, tab2 = st.tabs(["Data :clipboard:", "Analysis :weight_lifter:"])

models = ["Logistic Regression", "SVM", "Random Forests", "KNN", "Gausian Naive Bayes"]
all_ds = ["Iris", "Breast Cancer", "Wine"]
sel_ds = st.sidebar.selectbox("Select the dataset: ", all_ds)
base_models = st.sidebar.multiselect("Select the base models: ", models)
vo = st.sidebar.radio("Select the type of voting: ", ["hard", "soft"])
size = st.sidebar.slider("Select training size: ", 0.1, 0.9)
train = st.sidebar.button("Run Model")

# List to store base_estimators (base_models)
b_e = []

# Function to load and displayy dataset based on User Preference
def load_ds(sel_ds):
    if sel_ds == "Iris":
        ds = datasets.load_iris()
    elif sel_ds == "Breast Cancer":
        ds = datasets.load_breast_cancer()
    else:
        ds = datasets.load_wine()
    ds_df = pd.DataFrame(ds.data, columns=ds.feature_names)
    st.write(ds_df)
    return ds

# Storing base models as a tuple(str, base_model) inside a list called b_e(base_estimators)
def base_estimators(base_models):
    if "Logistic Regression" in base_models:
        b_e.append(('lr', LogisticRegression()))
    if "KNN" in base_models:
        b_e.append(('knn', KNeighborsClassifier()))
    if "Random Forests" in base_models:
        b_e.append(('rf', RandomForestClassifier()))
    if "SVM" in base_models:
        b_e.append(('svm', SVC(probability=True)))
    if "Gausian Naive Bayes" in base_models:
        b_e.append(('gnb', GaussianNB()))

base_estimators(base_models)

with tab1:
    ds = load_ds(sel_ds)
    X, y = ds.data, ds.target
    for b in b_e:
        x = cross_val_score(b[1],X,y,cv=10,scoring='accuracy')
        st.sidebar.write(b[0],np.round(np.mean(x),2))
    if b_e != []:
        vc = VotingClassifier(estimators=b_e, voting=vo)
        x = cross_val_score(vc,X,y,cv=10,scoring='accuracy', error_score='raise')
        st.sidebar.write(np.round(np.mean(x),2))



if train == True:
    X_train, X_test, Y_train, Y_test = train_test_split(ds.data, ds.target, train_size=size, random_state=123)
    vc.fit(X_train, Y_train)
    Y_test_preds = vc.predict(X_test)
    if vo == 'soft':
        Y_test_probs = vc.predict_proba(X_test)
    with tab2:
        # Classification Report
        st.write("#### Classification Report")
        st.code("=="+classification_report(Y_test, Y_test_preds, target_names=list(ds.target_names)))
        # Confusion Matrix
        st.write("#### Confusion Matrix")
        col1_a, col1_b = st.columns(2, gap="medium")
        with col1_a:
            conf_mat_fig = plt.figure(figsize=(6,6))
            ax1 = conf_mat_fig.add_subplot(111)
            skplt.metrics.plot_confusion_matrix(Y_test, Y_test_preds, ax=ax1)
            st.pyplot(conf_mat_fig, use_container_width=True)

        if vo == 'soft':
            st.write("#### ROC and Precision-Recall Graph")
            col2_a, col2_b = st.columns(2, gap="medium")
            with col2_a:
                roc_fig = plt.figure(figsize=(6,6))
                ax1 = roc_fig.add_subplot(111)
                skplt.metrics.plot_roc(Y_test, Y_test_probs, ax=ax1)
                st.pyplot(roc_fig, use_container_width=True)

            with col2_b:
                pr_fig = plt.figure(figsize=(6,6))
                ax1 = pr_fig.add_subplot(111)
                skplt.metrics.plot_precision_recall(Y_test, Y_test_probs, ax=ax1)
                st.pyplot(pr_fig, use_container_width=True)