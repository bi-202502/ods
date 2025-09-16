import streamlit as st
import requests

SDG_es_1 = """
    <div style="background-color:orange; padding: 25px; border-radius: 10px; text-align: center;">
        <h2 style="color:white;">Fin de la pobreza</h2>
        <a href="https://www.un.org/sustainabledevelopment/es/poverty/" target="_blank" style="color:white; font-size: 18px; text-decoration: underline;">
            Click aquí para saber más
        </a>
    </div>
    """

SDG_es_3 = """
    <div style="background-color:green; padding: 25px; border-radius: 10px; text-align: center;">
        <h2 style="color:white;">Salud y bienestar</h2>
        <a href="https://www.un.org/sustainabledevelopment/es/health/" target="_blank" style="color:white; font-size: 18px; text-decoration: underline;">
            Click aquí para saber más
        </a>
    </div>
    """

SDG_es_4 = """
    <div style="background-color:red; padding: 25px; border-radius: 10px; text-align: center;">
        <h2 style="color:white;">Educación de calidad</h2>
        <a href="https://www.un.org/sustainabledevelopment/es/education/" target="_blank" style="color:white; font-size: 18px; text-decoration: underline;">
            Click aquí para saber más
        </a>
    </div>
    """


def call_inference_api(text: str) -> int:
    try:
        response = requests.post(
            "http://localhost:8081/infer",
            json=[text],
            headers={"Content-Type": "application/json"},
        )
        response.raise_for_status()
        predictions = response.json()
        return predictions[0] if predictions else 1
    except Exception as e:
        st.error(f"Error calling inference API: {e}")
        return 1


def get_sdg_display(prediction: int) -> str:
    sdg_mapping = {1: SDG_es_1, 3: SDG_es_3, 4: SDG_es_4}
    return sdg_mapping.get(prediction, SDG_es_1)


def main():
    st.set_page_config(page_title="Metas ODS", page_icon="🌆")

    st.title("Metas de desarrollo sostenible")
    user_input = st.text_area("Brinda tus opiniones 🌐")

    if st.button("Analizar"):
        if user_input.strip():
            with st.spinner("Analizando tu texto..."):
                prediction = call_inference_api(user_input)
                st.session_state.prediction = prediction
                st.session_state.show_result = True
        else:
            st.warning("Por favor, ingresa un texto para analizar")

    if st.session_state.get("show_result", False):
        st.markdown("**Tu resultado es:**")
        prediction = st.session_state.get("prediction", 1)
        sdg_display = get_sdg_display(prediction)
        st.markdown(sdg_display, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
