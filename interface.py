import streamlit as st
import numpy as np
import pandas as pd
import scikitplot as skplt
import matplotlib.pyplot as plt
import seaborn as sns
import datetime as dt
import csv
import os

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, VotingClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.inspection import DecisionBoundaryDisplay

# User-Interface
st.set_page_config(layout="wide",
                   page_title="Ensemble+IDS Study",
                   page_icon="🤖")
st.write('<style>div.block-container{padding-top:2rem;}</style>', unsafe_allow_html=True)

st.title("Comparative analysis of Voting Classifiers over IDS Datasets")
st.write("### A joint study of Ensemble Models and IDS.")
tab1, tab2, tab3 = st.tabs(["Data :clipboard:", "Analysis :weight_lifter:", "Comparison :bar_chart:"])

models = ["Logistic Regression", "Random Forests", "KNN", "Gausian Naive Bayes", "Decision Tree", "AdaBoost"]
all_ds = ["KDDCup"]
sel_ds = st.sidebar.selectbox("Select the dataset: ", all_ds)
base_models = st.sidebar.multiselect("Select the base models: ", models)
vo = st.sidebar.radio("Select the type of voting: ", ["hard", "soft"])
size = st.sidebar.slider("Select training size: ", 0.1, 0.9)
train = st.sidebar.button("Run Model")

# List to store base_estimators (base_models)
b_e = []

# Storing data for conclusion
file_name = "performance_records.csv"
fields = ["Models", "Accuracy", "Precision", "Recall", "F1 Score", "Duration", "TP", "FP", "FN", "TN"]

print(os.path.exists(not('./'+file_name)))
if not(os.path.exists('./'+file_name)):
    with open(file_name, 'a', newline='') as csvfile:  
        # creating a csv writer object  
        csvwriter = csv.writer(csvfile)  
        # writing the fields  
        csvwriter.writerow(fields) 
    csvfile.close()

entries = []
values = []


def load_ds_for_display(sel_ds):
    if sel_ds == "KDDCup":
        ds_df = pd.read_csv('KDDCup_10percent.csv', header=None)
        ds = pd.read_csv('KDDCup_10percent.csv', header=None)
    st.write(ds_df)

def load_ds_for_model(sel_ds):
    if sel_ds == "KDDCup":
        df = pd.read_csv("KDDCup_10percent_preprocessed.csv")
    return df

# Storing base models as a tuple(str, base_model) inside a list called b_e(base_estimators)
def base_estimators(base_models):
    if "Logistic Regression" in base_models:
        b_e.append(('lr', LogisticRegression(solver='lbfgs', max_iter=1000)))
    if "KNN" in base_models:
        b_e.append(('knn', KNeighborsClassifier()))
    if "Random Forests" in base_models:
        b_e.append(('rf', RandomForestClassifier()))
    if "AdaBoost" in base_models:
        b_e.append(('adb', AdaBoostClassifier()))
    if "Gausian Naive Bayes" in base_models:
        b_e.append(('gnb', GaussianNB()))
    if "Decision Tree" in base_models:
        b_e.append(('dt', DecisionTreeClassifier()))

base_estimators(base_models)

with tab1:
    load_ds_for_display(sel_ds)
    ds_for_model = load_ds_for_model(sel_ds)
    if sel_ds == "KDDCup":
        y = ds_for_model['intrusion_binary']
        X = ds_for_model.drop('intrusion_binary', axis=1)
    for b in b_e:
        
        # start = dt.datetime.now()

        # x = cross_val_score(b[1],X,y,cv=10,scoring='accuracy')
        # st.sidebar.write(b[0],np.round(np.mean(x),2))

        # st.sidebar.write('Time taken:',dt.datetime.now()-start)
        
        vc = VotingClassifier(estimators=b_e, voting=vo)


    # if b_e != []:

    #     start = dt.datetime.now()

    #     vc = VotingClassifier(estimators=b_e, voting=vo)
    #     x = cross_val_score(vc,X,y,cv=10,scoring='accuracy', error_score='raise')
        
    #     st.sidebar.write(np.round(np.mean(x),2))

    #     st.sidebar.write('Time taken:',dt.datetime.now()-start)


if train == True:
    if sel_ds == "KDDCup":
        X_train, X_test, Y_train, Y_test = train_test_split(ds_for_model.drop('intrusion_binary', axis=1), ds_for_model['intrusion_binary'], stratify=ds_for_model['intrusion_binary'], train_size=size, random_state=123)    
    vc.fit(X_train, Y_train)
    start = dt.datetime.now()
    Y_test_preds = vc.predict(X_test)
    duration = str(dt.datetime.now()-start)
    if vo == 'soft':
        Y_test_probs = vc.predict_proba(X_test)
    with tab2:
        # Classification Report
        st.write("#### Classification Report")
        st.code("=="+classification_report(Y_test, Y_test_preds, target_names=list(['normal', 'malivolant'])))

        #CSV entry
        report = classification_report(Y_test, Y_test_preds, output_dict=True)

        cm  = confusion_matrix(Y_test, Y_test_preds)
        TP = cm[0][0]
        FP = cm[0][1]
        FN = cm[1][0]
        TN = cm[1][1]
        
        values.append(' '.join(([i[0] for i in b_e])))
        values.append(report['accuracy'])
        values.append(report['macro avg']['precision'])
        values.append(report['macro avg']['recall'])
        values.append(report['macro avg']['f1-score'])
        values.append(duration)
        values.append(TP)
        values.append(FP)
        values.append(FN)
        values.append(TN)
        entries.append(values)
        with open(file_name, 'a', newline='') as csvfile:  
            # creating a csv writer object  
            csvwriter = csv.writer(csvfile)     
            #csvwriter.writerow(fields)     
            # writing the data rows  
            csvwriter.writerows(entries)
        

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
        #with col2_a:
            # st.write('Best Model at present: ')
            # d = pd.read_csv('performance_records.csv')
            # best = d.loc[d['Accuracy'] == d.Accuracy.max()]
            # st.write(f"{best.Models.values}:")
            # st.write(f"Accuracy: {best.Accuracy.values}")
            # st.write(f"False Positive: {best.FP.values}")

plt.xticks(rotation=90)

if os.path.exists('./'+file_name):
    d = pd.read_csv('performance_records.csv')
    with tab3:        
        colt3_a, colt3_b = st.columns(2, gap='medium')
        with colt3_a:
            st.write("#### Accuracy Graph")
            plt.rc('xtick', labelsize=10) 
            plt.rc('ytick', labelsize=10)
            plt.xticks(rotation=90)
            acc = plt.figure(figsize=(6,6))
            #ax1 = acc.add_subplot(111)
            #plt.subplots(figsize=(2,2))
            sns.barplot(x="Models", y="Accuracy", width=0.3, data=d,palette='hot',edgecolor=sns.color_palette('dark',7))
            #plt.xticks(rotation=90)
            st.pyplot(acc, use_container_width=True)
            #st.pyplot(plt.gcf(), use_container_width=True)

            st.write("#### Precision Graph")
            plt.xticks(rotation=90)
            precs = plt.figure(figsize=(6,6))
            sns.barplot(x="Models", y="Precision", width=0.3, data=d,palette='hot',edgecolor=sns.color_palette('dark',7))
            st.pyplot(precs, use_container_width=True)

            st.write("#### Recall Graph")
            plt.xticks(rotation=90)
            rec = plt.figure(figsize=(6,6))
            sns.barplot(x="Models", y="Recall", width=0.3, data=d,palette='hot',edgecolor=sns.color_palette('dark',7))
            st.pyplot(rec, use_container_width=True)