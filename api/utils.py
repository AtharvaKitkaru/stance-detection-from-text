import pickle

import contractions
import nltk
import numpy as np
import tensorflow as tf
from gensim.models import KeyedVectors
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sentence_transformers import SentenceTransformer

nltk.download(["stopwords", "punkt"])
stopwords_english = stopwords.words("english")
operators = set(
    (
        "not",
        "against",
        "because",
        "until",
        "against",
        "between",
        "during",
        "into",
        "before",
        "after",
        "no",
        "nor",
        "won",
        "above",
        "down",
        "below",
        "over",
        "under",
    )
)

stopwords_english = set(stopwords_english) - operators

# GTR

with open("api/models/cf-a-u-nc.pkl", "rb") as f:
    cf_a_u_nc = pickle.load(f)

with open("api/models/cf-b-u-nc.pkl", "rb") as f:
    cf_b_u_nc = pickle.load(f)

with open("api/models/cf-a-u-gtr.pkl", "rb") as f:
    cf_a_u_gtr = pickle.load(f)

with open("api/models/cf-b-u-gtr.pkl", "rb") as f:
    cf_b_u_gtr = pickle.load(f)

with open("api/models/c-b-u-nc.pkl", "rb") as f:
    c_b_u_nc = pickle.load(f)

with open("api/models/c-b-u-gtr.pkl", "rb") as f:
    c_b_u_gtr = pickle.load(f)

with open("api/models/c-a-u-nc.pkl", "rb") as f:
    c_a_u_nc = pickle.load(f)

with open("api/models/c-a-u-gtr.pkl", "rb") as f:
    c_a_u_gtr = pickle.load(f)

with open("api/models/c-a-u-nc-fallback.pkl", "rb") as f:
    c_a_u_nc_fallback = pickle.load(f)

with open("api/models/c-a-u-gtr-fallback.pkl", "rb") as f:
    c_a_u_gtr_fallback = pickle.load(f)

with open("api/models/c-b-u-nc-fallback.pkl", "rb") as f:
    c_b_u_nc_fallback = pickle.load(f)

with open("api/models/c-b-u-gtr-fallback.pkl", "rb") as f:
    c_b_u_gtr_fallback = pickle.load(f)

# context-free
w2v = KeyedVectors.load("api/models/word2vec.bin")


def tokenize(text):
    text = contractions.fix(text.lower())
    return word_tokenize(text)


def encodeW2V(text):
    tokens = tokenize(text)
    encodings = []
    for token in tokens:
        if token in stopwords_english:
            continue
        else:
            try:
                encodings.append(w2v[token])
            except:
                continue
    return np.sum(encodings, axis=0)


# contextual
sentence_model = SentenceTransformer("all-distilroberta-v1")
sentence_embedding_dimension = sentence_model.get_sentence_embedding_dimension()
enc = lambda x: sentence_model.encode(x.numpy(), normalize_embeddings=True)


@tf.function
def encoder(x):
    encoded_x = tf.py_function(enc, inp=[x], Tout=tf.float32)
    encoded_x.set_shape((None, sentence_embedding_dimension))
    return encoded_x


# agglomerative
def _agglomerativeGTREncoder(x):
    encodings = enc(x)
    clusters = c_a_u_nc.predict(encodings)
    result = np.array(
        [
            c_a_u_gtr[cluster] if cluster != -1 else encoding
            for cluster, encoding in zip(clusters, encodings)
        ]
    )
    return result


@tf.function
def agglomerativeGTREncoder(x):
    encoded_x = tf.py_function(_agglomerativeGTREncoder, inp=[x], Tout=tf.float32)
    encoded_x.set_shape((None, sentence_embedding_dimension))
    return encoded_x


# bertopic
def _bertopicGTREncoder(x):
    encodings = enc(x)
    clusters = c_b_u_nc.predict(encodings)
    result = np.array(
        [
            c_b_u_gtr[cluster] if cluster != -1 else encoding
            for cluster, encoding in zip(clusters, encodings)
        ]
    )
    return result


@tf.function
def bertopicGTREncoder(x):
    encoded_x = tf.py_function(_bertopicGTREncoder, inp=[x], Tout=tf.float32)
    encoded_x.set_shape((None, sentence_embedding_dimension))
    return encoded_x


# !contextual fallback
sentence_model_fallback = SentenceTransformer("all-MiniLM-L6-v2")
sentence_embedding_dimension_fallback = (
    sentence_model_fallback.get_sentence_embedding_dimension()
)
enc_fallback = lambda x: sentence_model_fallback.encode(
    x.numpy(), normalize_embeddings=True
)


@tf.function
def encoderFallback(x):
    encoded_x = tf.py_function(enc_fallback, inp=[x], Tout=tf.float32)
    encoded_x.set_shape((None, sentence_embedding_dimension_fallback))
    return encoded_x


# agglomerative
def _agglomerativeGTREncoderFallback(x):
    encodings = enc_fallback(x)
    clusters = c_a_u_nc_fallback.predict(encodings)
    result = np.array(
        [
            c_a_u_gtr_fallback[cluster] if cluster != -1 else encoding
            for cluster, encoding in zip(clusters, encodings)
        ]
    )
    return result


@tf.function
def agglomerativeGTREncoderFallback(x):
    encoded_x = tf.py_function(
        _agglomerativeGTREncoderFallback, inp=[x], Tout=tf.float32
    )
    encoded_x.set_shape((None, sentence_embedding_dimension_fallback))
    return encoded_x


# bertopic
def _bertopicGTREncoderFallback(x):
    encodings = enc_fallback(x)
    clusters = c_b_u_nc_fallback.predict(encodings)
    result = np.array(
        [
            c_b_u_gtr_fallback[cluster] if cluster != -1 else encoding
            for cluster, encoding in zip(clusters, encodings)
        ]
    )
    return result


@tf.function
def bertopicGTREncoderFallback(x):
    encoded_x = tf.py_function(_bertopicGTREncoderFallback, inp=[x], Tout=tf.float32)
    encoded_x.set_shape((None, sentence_embedding_dimension_fallback))
    return encoded_x


# models
stanceMap = {0: "against", 1: "favor", 2: "neutral"}

modelFullName = {
    "gb": "Gradient Boosting Classifier",
    "svc": "Support Vector Classifier",
    "gnb": "Gaussian Naive Bayes Classifier",
    "lr": "Logistic Regression",
    "dt": "Decision Tree Classifier",
    "rf": "Random Forest Classifier",
    "knn": "K-Nearest Neighbors Classifier",
    "d": "Fully Connected Neural Network",
    "l": "LSTM Neural Network",
    "b": "Bi-LSTM Neural Network",
    "auto": "Auto",
}

typeOfModel = {
    "gb": "s",
    "svc": "s",
    "gnb": "s",
    "lr": "s",
    "dt": "s",
    "rf": "s",
    "knn": "s",
    "d": "n",
    "l": "n",
    "b": "n",
}


def predict(modelName, context, text, target, targetGrouping):
    result = []
    stance = "neutral"
    confidence = "Cannot determine"
    if modelName == "auto":

        # todo: cf-n-s-gb
        _ = "cf-n-s-gb"
        useModel = f"api/models/cf-n-s-gb.pkl"
        with open(useModel, "rb") as f:
            model = pickle.load(f)

        encoding = np.stack([encodeW2V(text + " " + target)])
        output = model.predict_proba(encoding)
        print(_, output)
        confidence = np.max(output)
        output = np.argmax(output, axis=1)[0]
        stance = stanceMap[output]
        # result.append((confidence, stance, _))
        # result.append({_: {"stance": stance, "confidence": str(confidence)}})
        result.append((_, stance, str(confidence)))

        # todo: cf-n-n-d
        _ = "cf-n-n-d"
        useModel = f"api/models/cf-n-n-d.h5"
        model = tf.keras.models.load_model(useModel)
        encoded_text = np.stack([encodeW2V(text)])
        encoded_target = np.stack([encodeW2V(target)])
        output = model.predict({"post": encoded_text, "topic": encoded_target})
        print(_, output)
        confidence = np.max(output)
        output = np.argmax(output, axis=1)[0]
        stance = stanceMap[output]
        # result.append((confidence, stance, _))
        # result.append({_: {"stance": stance, "confidence": str(confidence)}})
        result.append((_, stance, str(confidence)))

        # todo: cf-b-s-gb
        _ = "cf-b-s-gb"
        useModel = f"api/models/cf-b-s-gb.pkl"
        with open(useModel, "rb") as f:
            model = pickle.load(f)
        encoded_text = np.stack([encodeW2V(text)])
        encoded_target = np.stack([encodeW2V(target)])
        generalized_target = [cf_b_u_gtr[i] for i in cf_b_u_nc.predict(encoded_target)]
        output = model.predict_proba(
            np.concatenate((encoded_text, generalized_target), axis=-1)
        )
        print(_, output)
        confidence = np.max(output)
        output = np.argmax(output, axis=1)[0]
        stance = stanceMap[output]
        # result.append((confidence, stance, _))
        # result.append({_: {"stance": stance, "confidence": str(confidence)}})
        result.append((_, stance, str(confidence)))

        # todo: c-a-n-d
        _ = "c-a-n-d"
        useModel = f"api/models/c-a-n-d.h5"
        try:
            model = tf.keras.models.load_model(
                useModel,
                custom_objects={
                    "encoder": encoder,
                    "gtrEncoder": agglomerativeGTREncoder,
                },
            )
            output = model.predict(
                {"text": np.asarray([text]), "target": np.asarray([target])}
            )
        except:
            model = tf.keras.models.load_model(
                useModel,
                custom_objects={
                    "encoder": encoderFallback,
                    "gtrEncoder": agglomerativeGTREncoderFallback,
                },
            )
            output = model.predict(
                {"text": np.asarray([text]), "target": np.asarray([target])}
            )
        print(_, output)
        confidence = np.max(output)
        output = np.argmax(output, axis=1)[0]
        stance = stanceMap[output]
        # result.append((confidence, stance, _))
        # result.append({_: {"stance": stance, "confidence": str(confidence)}})
        result.append((_, stance, str(confidence)))

        # todo: c-b-n-b
        _ = "c-b-n-b"
        useModel = f"api/models/c-b-n-b.h5"
        try:
            model = tf.keras.models.load_model(
                useModel,
                custom_objects={
                    "encoder": encoder,
                    "gtrEncoder": bertopicGTREncoder,
                },
            )
            output = model.predict(
                {"text": np.asarray([text]), "target": np.asarray([target])}
            )
        except:
            model = tf.keras.models.load_model(
                useModel,
                custom_objects={
                    "encoder": encoderFallback,
                    "gtrEncoder": bertopicGTREncoderFallback,
                },
            )
            output = model.predict(
                {"text": np.asarray([text]), "target": np.asarray([target])}
            )
        print(_, output)
        confidence = np.max(output)
        output = np.argmax(output, axis=1)[0]
        stance = stanceMap[output]
        # result.append((confidence, stance, _))
        # result.append({_: {"stance": stance, "confidence": str(confidence)}})
        result.append((_, stance, str(confidence)))

    # ! manual
    else:
        # ! contextual
        if context == "c":
            if targetGrouping == "a":
                if typeOfModel[modelName] == "s":
                    useModel = f"api/models/c-a-s-{modelName}.pkl"
                    # sklearn
                    with open(useModel, "rb") as f:
                        c_a_s_sklearnModel = pickle.load(f)
                    try:
                        text_encoding = sentence_model.encode(
                            text, normalize_embeddings=True
                        )
                        target_encoding = sentence_model.encode(
                            target, normalize_embeddings=True
                        )
                        text_encoding = np.stack(text_encoding)
                        target_encoding = np.stack(target_encoding)
                        gtr = [
                            c_a_u_gtr[i] for i in c_a_u_nc.predict([target_encoding])
                        ]
                        inp = np.concatenate((text_encoding, gtr[0]), axis=-1)
                        if modelName == "svc":
                            output = c_a_s_sklearnModel.predict([inp])
                            stance = stanceMap[output[0]]
                        else:
                            output = c_a_s_sklearnModel.predict_proba([inp])
                            confidence = np.max(output)
                            output = np.argmax(output, axis=1)[0]
                            stance = stanceMap[output]
                    except:
                        text_encoding = sentence_model_fallback.encode(
                            text, normalize_embeddings=True
                        )
                        target_encoding = sentence_model_fallback.encode(
                            target, normalize_embeddings=True
                        )
                        text_encoding = np.stack(text_encoding)
                        target_encoding = np.stack(target_encoding)
                        gtr = [
                            c_a_u_gtr_fallback[i]
                            for i in c_a_u_nc_fallback.predict([target_encoding])
                        ]
                        inp = np.concatenate((text_encoding, gtr[0]), axis=-1)
                        if modelName == "svc":
                            output = c_a_s_sklearnModel.predict([inp])
                            stance = stanceMap[output[0]]
                        else:
                            output = c_a_s_sklearnModel.predict_proba([inp])
                            confidence = np.max(output)
                            output = np.argmax(output, axis=1)[0]
                            stance = stanceMap[output]
                else:
                    useModel = f"api/models/c-a-n-{modelName}.h5"
                    # tf
                    try:
                        model = tf.keras.models.load_model(
                            useModel,
                            custom_objects={
                                "encoder": encoder,
                                "gtrEncoder": agglomerativeGTREncoder,
                            },
                        )
                        output = model.predict(
                            {"text": np.asarray([text]), "target": np.asarray([target])}
                        )
                    except:
                        model = tf.keras.models.load_model(
                            useModel,
                            custom_objects={
                                "encoder": encoderFallback,
                                "gtrEncoder": agglomerativeGTREncoderFallback,
                            },
                        )
                        output = model.predict(
                            {"text": np.asarray([text]), "target": np.asarray([target])}
                        )
                    print(output)
                    confidence = np.max(output)
                    output = np.argmax(output, axis=1)[0]
                    stance = stanceMap[output]
            elif targetGrouping == "b":
                if typeOfModel[modelName] == "s":
                    useModel = f"api/models/c-b-s-{modelName}.pkl"
                    # sklearn
                    with open(useModel, "rb") as f:
                        c_b_s_sklearnModel = pickle.load(f)
                    try:
                        text_encoding = sentence_model.encode(
                            text, normalize_embeddings=True
                        )
                        target_encoding = sentence_model.encode(
                            target, normalize_embeddings=True
                        )
                        text_encoding = np.stack(text_encoding)
                        target_encoding = np.stack(target_encoding)
                        gtr = [
                            c_b_u_gtr[i] for i in c_b_u_nc.predict([target_encoding])
                        ]
                        inp = np.concatenate((text_encoding, gtr[0]), axis=-1)
                        if modelName == "svc":
                            output = c_b_s_sklearnModel.predict([inp])
                            stance = stanceMap[output[0]]
                        else:
                            output = c_b_s_sklearnModel.predict_proba([inp])
                            confidence = np.max(output)
                            output = np.argmax(output, axis=1)[0]
                            stance = stanceMap[output]
                    except:
                        text_encoding = sentence_model_fallback.encode(
                            text, normalize_embeddings=True
                        )
                        target_encoding = sentence_model_fallback.encode(
                            target, normalize_embeddings=True
                        )
                        text_encoding = np.stack(text_encoding)
                        target_encoding = np.stack(target_encoding)
                        gtr = [
                            c_b_u_gtr_fallback[i]
                            for i in c_b_u_nc_fallback.predict([target_encoding])
                        ]
                        inp = np.concatenate((text_encoding, gtr[0]), axis=-1)
                        if modelName == "svc":
                            output = c_b_s_sklearnModel.predict([inp])
                            stance = stanceMap[output[0]]
                        else:
                            output = c_b_s_sklearnModel.predict_proba([inp])
                            confidence = np.max(output)
                            output = np.argmax(output, axis=1)[0]
                            stance = stanceMap[output]
                else:
                    useModel = f"api/models/c-b-n-{modelName}.h5"
                    # tf
                    try:
                        model = tf.keras.models.load_model(
                            useModel,
                            custom_objects={
                                "encoder": encoder,
                                "gtrEncoder": bertopicGTREncoder,
                            },
                        )
                        output = model.predict(
                            {"text": np.asarray([text]), "target": np.asarray([target])}
                        )
                    except:
                        model = tf.keras.models.load_model(
                            useModel,
                            custom_objects={
                                "encoder": encoderFallback,
                                "gtrEncoder": bertopicGTREncoderFallback,
                            },
                        )
                        output = model.predict(
                            {"text": np.asarray([text]), "target": np.asarray([target])}
                        )
                    print(output)
                    confidence = np.max(output)
                    output = np.argmax(output, axis=1)[0]
                    stance = stanceMap[output]
            else:
                if typeOfModel[modelName] == "s":
                    useModel = f"api/models/c-n-s-{modelName}.pkl"
                    # sklearn
                    with open(useModel, "rb") as f:
                        c_n_s_sklearnModel = pickle.load(f)
                    try:
                        text_encoding = sentence_model.encode(
                            text, normalize_embeddings=True
                        )
                        target_encoding = sentence_model.encode(
                            target, normalize_embeddings=True
                        )
                        text_encoding = np.stack(text_encoding)
                        target_encoding = np.stack(target_encoding)
                        inp = np.concatenate((text_encoding, target_encoding), axis=-1)
                        if modelName == "svc":
                            output = c_n_s_sklearnModel.predict([inp])
                            stance = stanceMap[output[0]]
                        else:
                            output = c_n_s_sklearnModel.predict_proba([inp])
                            confidence = np.max(output)
                            output = np.argmax(output, axis=1)[0]
                            stance = stanceMap[output]
                    except:
                        text_encoding = sentence_model_fallback.encode(
                            text, normalize_embeddings=True
                        )
                        target_encoding = sentence_model_fallback.encode(
                            target, normalize_embeddings=True
                        )
                        text_encoding = np.stack(text_encoding)
                        target_encoding = np.stack(target_encoding)
                        inp = np.concatenate((text_encoding, target_encoding), axis=-1)
                        if modelName == "svc":
                            output = c_n_s_sklearnModel.predict([inp])
                            stance = stanceMap[output[0]]
                        else:
                            output = c_n_s_sklearnModel.predict_proba([inp])
                            confidence = np.max(output)
                            output = np.argmax(output, axis=1)[0]
                            stance = stanceMap[output]
                else:
                    useModel = f"api/models/c-n-n-{modelName}.h5"
                    # tf
                    try:
                        model = tf.keras.models.load_model(
                            useModel,
                            custom_objects={
                                "encoder": encoder,
                            },
                        )
                        output = model.predict(
                            {"text": np.asarray([text]), "target": np.asarray([target])}
                        )
                    except:
                        model = tf.keras.models.load_model(
                            useModel,
                            custom_objects={
                                "encoder": encoderFallback,
                            },
                        )
                        output = model.predict(
                            {"text": np.asarray([text]), "target": np.asarray([target])}
                        )
                    print(output)
                    confidence = np.max(output)
                    output = np.argmax(output, axis=1)[0]
                    stance = stanceMap[output]
        # !context-free
        else:
            if targetGrouping == "a":
                if typeOfModel[modelName] == "s":
                    useModel = f"api/models/cf-a-s-{modelName}.pkl"
                    with open(useModel, "rb") as f:
                        model = pickle.load(f)

                    encoded_text = np.stack([encodeW2V(text)])
                    encoded_target = np.stack([encodeW2V(target)])

                    generalized_target = [
                        cf_a_u_gtr[i] for i in cf_a_u_nc.predict(encoded_target)
                    ]
                    if modelName == "svc":
                        output = model.predict(
                            np.concatenate((encoded_text, generalized_target), axis=-1)
                        )
                        print(output)
                        stance = stanceMap[output[0]]
                    else:
                        output = model.predict_proba(
                            np.concatenate((encoded_text, generalized_target), axis=-1)
                        )
                        print(output)
                        confidence = np.max(output)
                        output = np.argmax(output, axis=1)[0]
                        stance = stanceMap[output]
                else:
                    useModel = f"api/models/cf-a-n-{modelName}.h5"
                    model = tf.keras.models.load_model(useModel)

                    encoded_text = np.stack([encodeW2V(text)])
                    encoded_target = np.stack([encodeW2V(target)])

                    generalized_target = np.stack(
                        [cf_a_u_gtr[i] for i in cf_a_u_nc.predict(encoded_target)]
                    )
                    output = model.predict(
                        {"post": encoded_text, "topic": generalized_target}
                    )
                    print(output)
                    confidence = np.max(output)
                    output = np.argmax(output, axis=1)[0]
                    stance = stanceMap[output]
            elif targetGrouping == "b":
                if typeOfModel[modelName] == "s":
                    useModel = f"api/models/cf-b-s-{modelName}.pkl"
                    with open(useModel, "rb") as f:
                        model = pickle.load(f)

                    encoded_text = np.stack([encodeW2V(text)])
                    encoded_target = np.stack([encodeW2V(target)])

                    generalized_target = [
                        cf_b_u_gtr[i] for i in cf_b_u_nc.predict(encoded_target)
                    ]
                    if modelName == "svc":
                        output = model.predict(
                            np.concatenate((encoded_text, generalized_target), axis=-1)
                        )
                        print(output)
                        stance = stanceMap[output[0]]
                    else:
                        output = model.predict_proba(
                            np.concatenate((encoded_text, generalized_target), axis=-1)
                        )
                        print(output)
                        confidence = np.max(output)
                        output = np.argmax(output, axis=1)[0]
                        stance = stanceMap[output]
                else:
                    useModel = f"api/models/cf-b-n-{modelName}.h5"
                    model = tf.keras.models.load_model(useModel)

                    encoded_text = np.stack([encodeW2V(text)])
                    encoded_target = np.stack([encodeW2V(target)])

                    generalized_target = np.stack(
                        [cf_b_u_gtr[i] for i in cf_b_u_nc.predict(encoded_target)]
                    )
                    output = model.predict(
                        {"post": encoded_text, "topic": generalized_target}
                    )
                    print(output)
                    confidence = np.max(output)
                    output = np.argmax(output, axis=1)[0]
                    stance = stanceMap[output]
            else:
                if typeOfModel[modelName] == "s":
                    useModel = f"api/models/cf-n-s-{modelName}.pkl"
                    with open(useModel, "rb") as f:
                        model = pickle.load(f)

                    encoding = np.stack([encodeW2V(text + " " + target)])

                    if modelName == "svc":
                        output = model.predict(encoding)
                        print(output)
                        stance = stanceMap[output[0]]
                    else:
                        output = model.predict_proba(encoding)
                        print(output)
                        confidence = np.max(output)
                        output = np.argmax(output, axis=1)[0]
                        stance = stanceMap[output]
                else:
                    useModel = f"api/models/cf-n-n-{modelName}.h5"
                    model = tf.keras.models.load_model(useModel)

                    encoded_text = np.stack([encodeW2V(text)])
                    encoded_target = np.stack([encodeW2V(target)])

                    output = model.predict(
                        {"post": encoded_text, "topic": encoded_target}
                    )
                    print(output)
                    confidence = np.max(output)
                    output = np.argmax(output, axis=1)[0]
                    stance = stanceMap[output]

    return stance, modelFullName[modelName], str(confidence), result
