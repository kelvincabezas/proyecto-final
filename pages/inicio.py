import streamlit as st

def show_inicio():
    # Título centrado
    st.markdown("<h2 style='text-align: center;'>DESARROLLO DE MODELOS PREDICTIVOS Y ESTRATEGIAS DE IMPLEMENTACIÓN PARA OPTIMIZAR LA PRODUCTIVIDAD Y RENTABILIDAD DE LA PESCA ARTESANAL EN EL PERÚ</h2>", unsafe_allow_html=True)

    # Espaciado adicional
    st.markdown("<br><br>", unsafe_allow_html=True)

    # Título de la introducción
    st.markdown("<h4 style='text-align: center;'>Introducción</h4>", unsafe_allow_html=True)

    # Primer párrafo
    st.markdown("""
        <div style='text-align: justify;'>
            La pesca artesanal en el Perú es una actividad fundamental que contribuye significativamente a la seguridad alimentaria y al sustento económico de numerosas comunidades costeras. Según el Ministerio de la Producción, en 2021 se estimaba que el número de pescadores artesanales superaba los 88,000 en todo el país, con aproximadamente 55,000 en el ámbito marítimo y 33,000 en el continental.
        </div><br>
    """, unsafe_allow_html=True)

    # Segundo párrafo
    st.markdown("""
        <div style='text-align: justify;'>
            La pesca artesanal aporta más del 64% de los productos marinos destinados al consumo humano directo, destacando especies como la pota, bonito, merluza, lisa, caballa y jurel.
        </div><br>
    """, unsafe_allow_html=True)

    # Tercer párrafo
    st.markdown("""
        <div style='text-align: justify;'>
            La Ley N° 31749, promulgada en mayo de 2023, reconoce la pesca tradicional ancestral y la pesca tradicional artesanal, impulsando su preservación dentro de las cinco millas marítimas peruanas.
        </div><br>
    """, unsafe_allow_html=True)

    # Cuarto párrafo
    st.markdown("""
        <div style='text-align: justify;'>
            En este contexto, el Ministerio de la Producción ha implementado el Programa Presupuestal 095: "Fortalecimiento de la Pesca Artesanal", que incluye acciones como la recopilación de información sobre volúmenes de descarga, especies y precios en diversos puntos de desembarque del litoral.
        </div><br>
    """, unsafe_allow_html=True)

    # Quinto párrafo
    st.markdown("""
        <div style='text-align: justify;'>
            Este proyecto tiene como objetivo desarrollar modelos predictivos y estrategias de implementación que optimicen la productividad y rentabilidad de la pesca artesanal en el Perú, basándose en datos recopilados a nivel nacional y considerando las particularidades de cada región.
        </div><br><br>
    """, unsafe_allow_html=True)

    # Título de Objetivos
    st.markdown("<h3 style='text-align: center;'>Objetivos</h3>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    # Objetivo General
    st.markdown("""
        **Objetivo General:**
        **Determinar las condiciones óptimas para la pesca artesanal en distintas regiones del país, evaluando factores como la hora de pesca y la relación entre talla y valor comercial de las especies.**
    """)

    # Objetivos específicos
    st.markdown("""
        1. Identificar los horarios de mayor productividad para la pesca artesanal en diferentes puntos de desembarque mediante modelos predictivos.
        
        2. Analizar la relación entre el tamaño de las capturas y el precio de mercado por kilogramo, diferenciando entre especies y zonas de pesca, empleando técnicas de regresión en Machine Learning.
        
        3. Comparar los volúmenes de captura a lo largo del día para identificar condiciones óptimas en diversas caletas, optimizando estrategias pesqueras.
        
        4. Validar un modelo predictivo que permita mejorar la rentabilidad y sostenibilidad de la pesca artesanal en función de variables ambientales y biológicas a nivel nacional.
    """)

    # Fuentes
    st.markdown("""
        <footer style='text-align: center; font-size: small; margin-top: 40px;'>
            Fuentes:
            <ul style="list-style-type: none;">
                <li><a href="https://www.gob.pe/institucion/produce/noticias/634552-produce-en-el-peru-hay-mas-de-88-000-pescadores-artesanales" target="_blank">Ministerio de la Producción, Pescadores Artesanales en el Perú</a></li>
                <li><a href="https://www.gob.pe/institucion/produce/noticias/189221-produce-pesca-artesanal-contribuye-con-mas-del-64-de-productos-marinos-de-chd" target="_blank">Ministerio de la Producción, Contribución de la Pesca Artesanal</a></li>
                <li><a href="https://www.gob.pe/institucion/produce/noticias/760799-gobierno-promulga-ley-que-reconoce-pesca-ancestral-y-tradicional-artesanal-y-su-preservacion-dentro-de-las-cinco-millas-maritimas" target="_blank">Gobierno promulga Ley N° 31749 sobre la Pesca Ancestral y Artesanal</a></li>
                <li><a href="https://ogeiee.produce.gob.pe/index.php/en/shortcode/oee-documentos-publicaciones/publicaciones-anuales/item/1001-anuario-estadisticoo-pesquero-y-acuicola-2020" target="_blank">Ministerio de la Producción, Programa Presupuestal 095: Fortalecimiento de la Pesca Artesanal</a></li>
            </ul>
        </footer>
    """, unsafe_allow_html=True)