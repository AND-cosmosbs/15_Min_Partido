import streamlit as st

st.set_page_config(page_title="HT/FT Strategy", layout="wide")

st.title("Estrategia HT/FT – Versión 0.1")

st.write("""
Esta es la primera versión de la app, desplegada solo para comprobar que 
todo el pipeline GitHub → Render → Streamlit funciona correctamente.
""")

st.header("Checklist de despliegue")
st.markdown("""
- ✅ Repo creado en GitHub  
- ✅ Estructura de carpetas (`app/`, `backend/`, `data/`, `utils/`)  
- ✅ `requirements.txt` añadido  
- ✅ `app/main.py` creado  
- ⏳ Conexión con Supabase (próximo paso)  
- ⏳ Cargar fixtures y puntuarlos con el modelo  
- ⏳ Pantalla para ver lista de partidos recomendados  
""")
