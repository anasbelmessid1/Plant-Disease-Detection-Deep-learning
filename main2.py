import streamlit as st
import tensorflow as tf
import numpy as np
import openai
from threading import Thread

# Configuration de l'API OpenAI
openai.api_key = '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'
max_tokens = 4096  # Define max_tokens at the beginning of the script

def translate_and_recommend(disease_name):
    prompt_text = f"""
    Vous êtes un assistant agricole hautement qualifié, formé en pathologie végétale et parfaitement bilingue en anglais et en français. 
    Votre tâche consiste à fournir des conseils d'expert en français sur la gestion des maladies des plantes, 
    en intégrant des pratiques agronomiques avancées, des recherches actuelles et des directives agricoles locales.

    J'ai détecté une maladie appelée '{disease_name}' dans nos cultures. 
    Pourriez-vous traduire le nom de cette maladie en français et fournir des recommandations détaillées et expertes sur la manière de gérer efficacement cette maladie?
    """
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": prompt_text},
            {"role": "user", "content": "Veuillez procéder avec vos conseils."}
        ]
    )
    return response.choices[0].message['content'].strip()

def chat_with_openai(user_input, chat_log=None):
    if chat_log is None:
        chat_log = []
    chat_log.append({"role": "user", "content": user_input})
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=chat_log
    )
    chat_log.append({"role": "assistant", "content": response.choices[0].message['content']})
    return response.choices[0].message['content'], chat_log

def model_prediction(test_image):
    model = tf.keras.models.load_model('/content/drive/MyDrive/Plant-Disease-Detection/Plant_Disease_Detection/trained_plant_disease_model.keras')
    image = tf.keras.preprocessing.image.load_img(test_image, target_size=(128,128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])
    predictions = model.predict(input_arr)
    return np.argmax(predictions) 

# Configuration de la barre latérale
st.sidebar.title("Tableau de bord")
app_mode = st.sidebar.selectbox("Choisir une page", ["Accueil", "À propos", "Reconnaissance des maladies", "Assistant Chatbot"])

# Page d'accueil
if app_mode == "Accueil":
    st.header("Système de Reconnaissance des Maladies des Plantes")
    image_path = "/content/drive/MyDrive/Plant-Disease-Detection/Plant_Disease_Detection/homepage.jpg"
    st.image(image_path, use_column_width=True)
    st.markdown("""
    Bienvenue dans notre système de reconnaissance des maladies des plantes! 🌿🔍
    
    Notre mission est d'aider à identifier efficacement les maladies des plantes en utilisant des pratiques agricoles durables. Téléchargez une image d'une plante, et notre système l'analysera pour détecter tout signe de maladies. Ensemble, protégeons nos cultures et assurons une récolte plus saine!

    ### Comment ça fonctionne
    1. **Télécharger une image :** Rendez-vous sur la page **Reconnaissance des maladies** et téléchargez une image d'une plante suspectée de maladies.
    2. **Analyse :** Notre système traitera l'image en utilisant des algorithmes avancés pour identifier les maladies potentielles.
    3. **Résultats :** Consultez les résultats et les recommandations pour des actions ultérieures.
    """)

# Page À propos
elif app_mode == "À propos":
    st.header("À propos")
    st.markdown("""
    #### À propos des données
    Ces données ont été recréées à partir d'un ensemble original en utilisant une augmentation hors ligne. Le jeu de données original est disponible sur ce dépôt GitHub.
    Ce jeu comprend environ 87 000 images RGB de feuilles de cultures saines et malades, classées en 38 catégories différentes. L'ensemble total des données est divisé en une proportion de 80/20 entre les ensembles d'entraînement et de validation, en préservant la structure des répertoires.
    Un nouveau répertoire contenant 33 images de test a été créé ultérieurement à des fins de prédiction.
    """)

# Page de reconnaissance des maladies
elif app_mode == "Reconnaissance des maladies":
    st.header("Reconnaissance des maladies")
    test_image = st.file_uploader("Choisissez une image :", key="file_uploader")

    if 'display_image' not in st.session_state:
        st.session_state.display_image = False

    if test_image is not None:
        if st.button("Afficher l'image"):      
            st.session_state.display_image = not st.session_state.display_image

        if st.session_state.display_image:
            st.image(test_image, use_column_width=True)

        if st.button("Prédire"):
            st.session_state.display_image = False
            with st.spinner("Analyse de l'image en cours..."):
                class_name = [
                    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
                    'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 
                    'Cherry_(including_sour)___healthy', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 
                    'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 
                    'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 
                    'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot',
                    'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 
                    'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy', 
                    'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew', 
                    'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot', 
                    'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold', 
                    'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 
                    'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus',
                    'Tomato___healthy'
                ]
                result_index = model_prediction(test_image)
                if result_index >= len(class_name):
                    st.error("L'indice de prédiction dépasse la plage. Veuillez vérifier le modèle et les noms des classes.")
                else:
                    predicted_disease = class_name[result_index]
                    st.success(f"Prédiction du modèle : {predicted_disease}")
                    
                    try:
                        translation_and_recommendation = translate_and_recommend(predicted_disease)
                        st.markdown("### Traduction et Recommandation")
                        st.write(translation_and_recommendation)
                    except Exception as e:
                        st.error(f"Échec de l'obtention de la traduction et des recommandations : {e}")

    else:
        st.warning("Veuillez télécharger une image pour poursuivre la reconnaissance de la maladie.")

# Page de l'Assistant Chatbot
elif app_mode == "Assistant Chatbot":
    st.title("🤖 Assistant Chatbot pour la Gestion des Maladies des Plantes 🌿")

    # Initialize the full chat messages history for UI
    if "full_chat_history" not in st.session_state:
        st.session_state["full_chat_history"] = [{"role": "system", "content": "Je suis un chatbot informatif qui assiste dans la gestion des maladies des plantes."}]

    # Initialize the API chat messages history for OpenAI requests
    if "api_chat_history" not in st.session_state:
        st.session_state["api_chat_history"] = [{"role": "system", "content": "Je suis un chatbot informatif qui assiste dans la gestion des maladies des plantes."}]

    # Input for new user messages
    if (prompt := st.chat_input("Posez-moi une question sur les maladies des plantes et leur gestion :")) is not None:
        st.session_state.full_chat_history.append({"role": "user", "content": prompt})

        # Limit the number of messages sent to OpenAI by token count
        total_tokens = sum(len(message["content"]) for message in st.session_state["api_chat_history"])
        token_buffer = 500

        while total_tokens + len(prompt) + token_buffer > max_tokens:
            removed_message = st.session_state["api_chat_history"].pop(0)
            total_tokens -= len(removed_message["content"])

        st.session_state.api_chat_history.append({"role": "user", "content": prompt})

    # Display previous chat messages from full_chat_history (ignore system prompt message)
    for message in st.session_state["full_chat_history"][1:]:
        if message["role"] == "user":
            st.chat_message("user", avatar='🧑‍💻').write(message["content"])
        elif message["role"] == "assistant":
            st.chat_message("assistant", avatar='🤖').write(message["content"])

    # Check if the last message in api_chat_history is from the user
    if st.session_state["api_chat_history"][-1]["role"] != "assistant":
        with st.spinner("⌛Connexion au modèle AI..."):
            # Send only the most recent messages to OpenAI from api_chat_history
            recent_messages = st.session_state["api_chat_history"][-5:]  # Limit to the last 5 messages
            new_message, st.session_state["api_chat_history"] = chat_with_openai(recent_messages[-1]["content"], st.session_state["api_chat_history"])

            # Add this latest message to both api_chat_history and full_chat_history
            st.session_state["api_chat_history"].append({"role": "assistant", "content": new_message})
            st.session_state["full_chat_history"].append({"role": "assistant", "content": new_message})

            # Display the latest message from the assistant
            st.chat_message("assistant", avatar='🤖').write(new_message)

    # Display token usage and progress
    current_tokens = sum(len(message["content"]) for message in st.session_state["full_chat_history"])
    progress = min(1.0, max(0.0, current_tokens / max_tokens))
    st.progress(progress)
    st.write(f"Tokens Utilisés: {current_tokens}/{max_tokens}")
    if current_tokens > max_tokens:
        st.warning("Note : En raison des limites de caractères, certains anciens messages pourraient ne pas être pris en compte dans les conversations en cours avec l'IA.")
