# Hyperspectral Crop Classification - Indian Pines (demo app)
# Choose a feature set and a model, then see the metrics and the classified map

from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score, confusion_matrix, classification_report

st.set_page_config(page_title="Hyperspectral Classification - Indian Pines", layout="wide")

FOLDER = Path(__file__).parent

# Step 1: Class names and map colours (0 = background, no label)
class_names = ['Alfalfa', 'Corn-notill', 'Corn-mintill', 'Corn', 'Grass-pasture',
               'Grass-trees', 'Grass-pasture-mowed', 'Hay-windrowed', 'Oats',
               'Soybean-notill', 'Soybean-mintill', 'Soybean-clean', 'Wheat',
               'Woods', 'Buildings-Grass-Trees-Drives', 'Stone-Steel-Towers']
my_cmap = ListedColormap(['white'] + [plt.cm.tab20(i) for i in range(16)])


# Step 2: Load the data and prepare it (same steps as the notebook)
@st.cache_resource
def prepare():
    img = np.load(FOLDER /"archive" / 'indianpinearray.npy')
    gt = np.load(FOLDER /"archive" / 'IPgt.npy')

    # If the file still has 220 bands, remove the water absorption bands
    if img.shape[2] == 220:
        bad_bands = list(range(103, 108)) + list(range(149, 163)) + [219]
        img = np.delete(img, bad_bands, axis=2)

    X_all = img.reshape(-1, img.shape[2])
    y_all = gt.reshape(-1)

    # Keep only labelled pixels (remove background = 0)
    X = X_all[y_all > 0]
    y = y_all[y_all > 0]

    # 80% training and 20% testing, same split as the notebook
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=1, stratify=y)

    # Scale the bands (fitted on training data only)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    X_lab_s = scaler.transform(X)

    return {'gt': gt, 'X_train': X_train, 'y_train': y_train, 'y_test': y_test,
            'X_train_s': X_train_s, 'X_test_s': X_test_s, 'X_lab_s': X_lab_s}


# Step 3: PCA (fitted on the scaled training data)
@st.cache_resource
def get_pca():
    d = prepare()
    pca = PCA()
    train_p = pca.fit_transform(d['X_train_s'])
    test_p = pca.transform(d['X_test_s'])
    lab_p = pca.transform(d['X_lab_s'])
    return train_p, test_p, lab_p


# Step 4: Mutual Information scores (uses mi_scores.npy if it is in the folder)
@st.cache_resource
def get_mi_scores():
    path = FOLDER / 'mi_scores.npy'
    if path.exists():
        return np.load(path)
    d = prepare()
    return mutual_info_classif(d['X_train'], d['y_train'], random_state=1)


# Step 5: Order of bands for "MI + redundancy penalty" (simple mRMR style)
@st.cache_resource
def redundancy_order(max_bands=100):
    d = prepare()
    mi_scores = get_mi_scores()
    corr = np.abs(np.corrcoef(d['X_train_s'], rowvar=False))
    mi_norm = mi_scores / mi_scores.max()

    selected = [int(np.argmax(mi_scores))]
    remaining = [b for b in range(d['X_train_s'].shape[1]) if b != selected[0]]
    while len(selected) < max_bands:
        scores = [mi_norm[b] - corr[b, selected].mean() for b in remaining]
        best = remaining[int(np.argmax(scores))]
        selected.append(best)
        remaining.remove(best)
    return selected


# Step 6: Build the features for the chosen method
def get_features(method, n):
    d = prepare()
    n_bands = d['X_train_s'].shape[1]

    if method == 'All bands':
        return d['X_train_s'], d['X_test_s'], d['X_lab_s'], 'All ' + str(n_bands) + ' bands'

    if method == 'PCA':
        train_p, test_p, lab_p = get_pca()
        return train_p[:, :n], test_p[:, :n], lab_p[:, :n], 'First ' + str(n) + ' principal components'

    if method == 'Evenly spaced bands':
        cols = np.linspace(0, n_bands - 1, n).astype(int)
    elif method == 'MI + redundancy penalty':
        cols = np.array(redundancy_order()[:n])
    else:  # 'Mutual Information top-N'
        cols = np.argsort(get_mi_scores())[::-1][:n]

    info = 'Selected bands (1 to ' + str(n_bands) + '): ' + ', '.join(str(c + 1) for c in sorted(cols))
    return d['X_train_s'][:, cols], d['X_test_s'][:, cols], d['X_lab_s'][:, cols], info


# Step 7: Models (same settings as the notebook)
def make_model(name):
    if name == 'SVM':
        return SVC(kernel='rbf', C=100, gamma='scale')
    if name == 'Random Forest':
        return RandomForestClassifier(n_estimators=200, random_state=1)
    return MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=300, early_stopping=True, random_state=1)


# Step 8: Train, test and make the classified map
@st.cache_data(show_spinner=False)
def run_model(method, n, model_name):
    d = prepare()
    Xtr, Xte, Xlab, info = get_features(method, n)

    model = make_model(model_name)
    model.fit(Xtr, d['y_train'])
    pred = model.predict(Xte)
    y_test = d['y_test']

    cm = confusion_matrix(y_test, pred, labels=list(range(1, 17)))
    report = classification_report(y_test, pred, labels=list(range(1, 17)), target_names=class_names,
                                   output_dict=True, zero_division=0)
    per_class = pd.DataFrame(report).T.iloc[:16][['precision', 'recall', 'f1-score', 'support']].round(3)
    per_class['support'] = per_class['support'].astype(int)

    # Classified map (labelled pixels only)
    gt = d['gt']
    pred_map = np.zeros(gt.shape, dtype=int)
    pred_map[gt > 0] = model.predict(Xlab)

    return {'oa': accuracy_score(y_test, pred), 'kappa': cohen_kappa_score(y_test, pred),
            'f1': f1_score(y_test, pred, average='macro'), 'cm': cm, 'per_class': per_class,
            'pred_map': pred_map, 'info': info, 'n_features': Xtr.shape[1], 'n_train': len(Xtr), 'n_test': len(Xte)}


# Step 9: Page layout
st.title("Hyperspectral Crop Classification - Indian Pines")
st.write("AVIRIS Indian Pines scene (Indiana, USA): 145 x 145 pixels, 200 bands, 16 land-cover classes. "
         "Choose the features and the model, then click Run to see the test metrics and the classified map.")

tab1, tab2 = st.tabs(["Try a model", "Study results"])

with tab1:
    st.sidebar.header("Choose features and model")
    method = st.sidebar.radio("Features", ['All bands', 'Evenly spaced bands', 'MI + redundancy penalty',
                                           'Mutual Information top-N', 'PCA'])
    n = 0
    if method == 'PCA':
        n = st.sidebar.slider("Number of components", 3, 100, 38)
    elif method != 'All bands':
        n = st.sidebar.slider("Number of bands", 5, 100, 50, step=5)
    model_name = st.sidebar.selectbox("Model", ['SVM', 'Neural Network (MLP)', 'Random Forest'])
    run = st.sidebar.button("Run")

    if run:
        with st.spinner("Training the model (this can take up to a minute)..."):
            st.session_state['hs_result'] = run_model(method, n, model_name)
            st.session_state['hs_title'] = model_name + " | " + method + (" (" + str(n) + ")" if n else "")

    if 'hs_result' not in st.session_state:
        st.info("Choose the features and model in the left panel, then click Run.")
    else:
        res = st.session_state['hs_result']
        st.subheader(st.session_state['hs_title'])
        st.caption(res['info'])

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Overall Accuracy", round(res['oa'], 4))
        c2.metric("Kappa", round(res['kappa'], 4))
        c3.metric("F1 (macro)", round(res['f1'], 4))
        c4.metric("Features used", res['n_features'])
        st.caption("Metrics are calculated on " + str(res['n_test']) + " test pixels. "
                   "The model was trained on " + str(res['n_train']) + " pixels.")

        gt = prepare()['gt']
        col1, col2 = st.columns(2)
        with col1:
            fig, ax = plt.subplots(figsize=(5, 5))
            ax.imshow(gt, cmap=my_cmap, vmin=0, vmax=16)
            ax.set_title('Ground truth')
            st.pyplot(fig)
        with col2:
            fig, ax = plt.subplots(figsize=(5, 5))
            ax.imshow(res['pred_map'], cmap=my_cmap, vmin=0, vmax=16)
            ax.set_title('Classified map')
            st.pyplot(fig)
        st.caption("The classified map includes the pixels used for training, so it looks better than the real "
                   "performance. Use the test metrics above for accuracy.")

        col3, col4 = st.columns(2)
        with col3:
            st.write("Confusion matrix (rows = true class, columns = predicted class)")
            cm = res['cm']
            fig, ax = plt.subplots(figsize=(6.5, 6))
            ax.imshow(cm, cmap='Blues')
            ax.set_xticks(range(16))
            ax.set_xticklabels(range(1, 17), fontsize=7)
            ax.set_yticks(range(16))
            ax.set_yticklabels(range(1, 17), fontsize=7)
            for i in range(16):
                for j in range(16):
                    if cm[i, j] > 0:
                        ax.text(j, i, cm[i, j], ha='center', va='center', fontsize=5,
                                color='white' if cm[i, j] > cm.max() * 0.5 else 'black')
            ax.set_xlabel('Predicted class')
            ax.set_ylabel('True class')
            st.pyplot(fig)
        with col4:
            st.write("Score for each class (number in front = class ID)")
            table = res['per_class'].copy()
            table.index = [str(i + 1) + ' ' + name for i, name in enumerate(table.index)]
            st.dataframe(table, height=600)
            st.caption("Classes 1, 7 and 9 have only 4 to 9 test pixels, so their scores change a lot with one error.")

with tab2:
    st.write("Results from the notebook study (one random 80/20 split, test set of 2,050 pixels).")
    final_path = FOLDER / 'results_final_comparison.csv'
    bands_path = FOLDER / 'results_band_methods.csv'

    if final_path.exists():
        st.subheader("All feature sets and models, best first")
        st.dataframe(pd.read_csv(final_path))
    else:
        st.info("Add results_final_comparison.csv to the app folder to show the study table here.")

    if bands_path.exists():
        st.subheader("Accuracy against number of bands (SVM)")
        bands = pd.read_csv(bands_path).set_index('Bands')
        st.line_chart(bands)
        st.dataframe(bands)

    st.subheader("Limits to keep in mind")
    st.write("- The train-test split is a random split of pixels, as in most papers on this dataset. Neighbouring pixels "
             "in the same field look almost identical, so accuracy is optimistic compared with splitting by area.")
    st.write("- Only one split was run, so differences of about one percentage point may be noise.")
    st.write("- The three smallest classes (Oats, Grass-pasture-mowed, Alfalfa) have very few test pixels.")
    st.write("- Neural network results change slightly from run to run.")
    st.write("Data: AVIRIS Indian Pines scene, a public benchmark dataset.")
