import pandas as pd
import joblib

onehot = joblib.load("fichs/onehot.joblib")
colonne_hot = joblib.load("fichs/colonne_hot.joblib")
model = joblib.load("fichs/model_final.joblib")


def predict(dt):
    # transform de onehotencoder
    cat_trans = onehot.transform(dt[colonne_hot])
    cols = onehot.get_feature_names_out(colonne_hot)
    new = pd.DataFrame(cat_trans, columns=cols)
    resultat = pd.concat([new, dt], axis=1)
    resultat = resultat.drop(columns=['spectral_type'], axis=1)
   # print(resultat)

    cls_pred = model.predict_proba(resultat)

    return cls_pred
