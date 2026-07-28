import streamlit as st
import joblib
import pandas as pd
from exploi_model import predict

st.set_page_config(
    page_title='Predict Class Stelaire',
    layout='wide',
    page_icon='✨'
)

st.title("Genesis Hitech IA Tool And Data Science Innovation", text_alignment='center')

# nom du site
st.title("Predict Class Stelaire ✨", text_alignment='center')

st.markdown("---")


st.markdown('## Choisir la valeur des variables', text_alignment='center')
# premiere ligne de 4 colonnes
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown('#### Alplha')
    slice_alpha = st.slider(
        label='Alpha', 
        min_value=0.000, 
        max_value=360.000, 
        label_visibility='collapsed',
    )

with col2:
    st.markdown('#### Delta')
    slide_delta = st.slider(
        label='delta', 
        min_value=-15.00,
        max_value=80.00, 
        label_visibility='collapsed'
    )

with col3:
    st.markdown('#### U')
    slide_u = st.slider(
        label='U',
        min_value=-1.00,
        max_value=29.00,
        label_visibility='collapsed'
    )

with col4:
    st.markdown('#### G')
    slide_g = st.slider(
        label='G',
        min_value=13.00,
        max_value=28.00,
        label_visibility='collapsed'
    )

# ligne 2

col5, col6, col7, col8 = st.columns(4)
with col5:
    st.markdown('#### R')
    slide_r = st.slider(
        label='R',
        min_value= 12.00,
        max_value=26.00,
        label_visibility='collapsed'
    )

with col6:
    st.markdown('#### I')
    slide_i = st.slider(
        label='i',
        min_value=11.00,
        max_value=28.00,
        label_visibility='collapsed'
    )

with col7:
    st.markdown('#### Z')
    slide_z = st.slider(
        label='z',
        min_value=11.00,
        max_value=28.00,
        label_visibility='collapsed'
    )

with col8:
    st.markdown('#### Redshift')
    slide_redshift = st.slider(
        label='Redshift',
        min_value=-1.00,
        max_value=8.00,
        label_visibility='collapsed'
    )

# ligne 3 variables categorielles

col9, col10, col11, col12 = st.columns(4)

with col9:
    st.markdown('#### Spectral_type')
    spectral_type_option = st.selectbox(
        'Selectionner le type spectral',
        ['M', 'O/B', 'G/K', 'A/F']
    )
with col10:
    pass
with col11:
    pass
with col12:
    st.markdown('#### Galaxy_polulation')
    galaxy_population = st.selectbox(
        'Selectionner la population galaxy',
        ['Red_sequence', 'Blue_cloud']
    )

# les classes a predire

st.markdown('#### Les classes stelaire a predire sont precisement : Galaxy, QSO egt STAR', text_alignment='center')

# bouton de prediction
left, middle, right = st.columns(3)
middle.markdown("Invidation a utiliser la demo", text_alignment='center')
btn = middle.button('Demo', width="stretch", icon='🚀')


# resuktat de la prediction
colg, colq, cols = st.columns(3)
if btn is True:
    d = {
        'spectral_type': spectral_type_option,
        'alpha': slice_alpha,
        'delta': slide_delta,
        'u': slide_u,
        'g': slide_g,
        'r': slide_r,
        'i': slide_i,
        'z': slide_z,
        'redshift': slide_redshift,
        'galaxy_population': 0 if galaxy_population == "Red_sequence" else 1
    }
    data = pd.DataFrame(d, index=[0])
    print(data)
    scores = predict(data)
    print(scores)

    with colg:
        st.markdown(f'#### Galaxy score : {scores[0][0]:.2f}')
        st.progress(float(scores[0][0]))
        cold, colm, colf = st.columns([1,5,1])
        cold.write("0%")
        colm.write("")
        colf.write("100%")

    with colq:
        st.markdown(f'#### QSO score : {scores[0][1]:.2f}')
        st.progress(float(scores[0][1]))
        cold, colm, colf = st.columns([1,5,1])
        cold.write("0%")
        colm.write("")
        colf.write("100%")

    with cols:
        st.markdown(f'#### STAR score : {scores[0][2]:.2f}')
        st.progress(float(scores[0][2]))
        cold, colm, colf = st.columns([1,5,1])
        cold.write("0%")
        colm.write("")
        colf.write("100%")


st.markdown("---")

st.subheader("Information sur les variables et les resultats du modele", text_alignment='center')

col1, col2, col3 = st.columns([1,3,1])

with col2:
    tab1, tab2, tab3, tab4 = st.tabs([
        "Relation Multivarier",
        "Degré d'importence avec la var cible",
        "Matrice de Confusion",
        "classification report"
    ], width='stretch')

with tab1:
    col1, col2 = st.columns(2)
    with col1:
        st.image('img\importation_multivar_num.png')
        with st.expander("Explication"):
            st.markdown("""
                ceux graphique montre la relation entre les variables
                quantitatives dans notre jeux de donnée. 
                le test statistique utiliser est le test de ***spearman*** qui est un test paramétrique
            """)
    with col2:
        st.image("img\importation_multivar_categorielle.png")
        with st.expander("Explication"):
            st.markdown("""
                ceux graphique montre l'intensitée de la relation entre les variables
                qualitatives dans notre jeux de donnée. 
                le test statistique utiliser est le test de ***chi2*** et ****le V de Cramer*** qui est un test paramétrique
            """)

with tab2:
    col1, col2 = st.columns(2)
    with col1:
        st.image('img\importation_var_numerique.png')
        with st.expander('Explication'):
            st.markdown("""
                ce graphique montre le degré d'importance des 
                variables sur la variable cible. le test statistique
                utiliser ici est le test de ***Kruskall-wallis*** qui est un test non paramétrique
            """)
    with col2:
        st.image("img\importation_var_categorielle.png")
        with st.expander('Explication'):
            st.markdown("""
                ce graphique montre le degré d'importance des 
                variables sur la variable cible. le test statistique
                utiliser ici est le test de ***de CHi2 et le V de Cramer***\n
                ***CHI2*** pour identifier la relation et ***V de Cramer*** pour l'intensité de la relation.
            """)

with tab3:
    col1, col2 = st.columns(2)
    with col1:
        st.image("img\matrice_cofusion_train.png", caption="Matrice de confusion pour les données d'entrainement")
    with col2:
        st.image("img\matrice_cofusion_test.png", caption="Matrice de confusion pour les données de test")

with tab4:

    st.text("Classification report d'entrainement")
    confusion_matrix = pd.DataFrame({
        "Precision": [0.99, 0.97, 0.91],
        "Recall": [0.97, 0.99, 0.99],
        "F1-score": [0.98, 0.97, 0.95],
        "Support": [301786, 93900, 66191],
    },index=["Galaxy", "QSO", "Star"],)
    st.dataframe(confusion_matrix)
    st.write('accuracy : 0.97')
    
    st.text("classification report de test")
    confusion_matrix = pd.DataFrame({
        "Precision": [0.98, 0.95, 0.88],
        "Recall": [0.96, 0.97, 0.96],
        "F1-score": [0.97, 0.96, 0.92],
        "Support": [75694, 23243, 16533],
    },   index=["Galaxy", "QSO", "Star"],)
    st.dataframe(confusion_matrix)
    st.write('accuracy : 0.96')

st.subheader("FAIRE PAR KEUTCHA TOALEU JOEL DATA SCIENTIST AND IA ENGENIER", text_alignment='center')

st.markdown("<h6 style='text-align: center;'>&copy; 2026 predict Classe Stelaire. Tous droit reserves.</h6>", unsafe_allow_html=True)