import streamlit as st
import pandas as pd
from supabase import create_client, Client
import openpyxl
from datetime import datetime, time
from dateutil import parser
import re


def show_carga():
    # Título de la aplicación
    st.subheader("REGISTRO DEL DESEMBARQUE DE RECURSOS HIDROBIOLÓGICOS PROCEDENTE DEL ÁMBITO MARÍTIMO")

    # Función para conectar con Supabase
    @st.cache_resource
    def get_supabase_client() -> Client:
        url = st.secrets["connections"]["supabase"]["SUPABASE_URL"]
        key = st.secrets["connections"]["supabase"]["SUPABASE_KEY"]
        return create_client(url, key)

    supabase = get_supabase_client()

    # Función para extraer el lugar sin la fecha usando regex
    def extraer_lugar(lugar_de_desembarque):
        """
        Extrae solo la parte alfabética del lugar de desembarque, ignorando cualquier parte numérica.
        Por ejemplo:
            'PARACHIQUE 01012024' -> 'PARACHIQUE'
            'CALETA COISCHCO01012024' -> 'CALETA COISCHCO'
            'COISHCO11112024' -> 'COISHCO'
        """
        match = re.match(r'([A-Za-z\s]+)', lugar_de_desembarque)
        if match:
            return match.group(1).strip().upper()
        else:
            # Si no coincide con la regex, devolver el valor original en mayúsculas
            return lugar_de_desembarque.strip().upper()

    # Sección para cargar archivos Excel
    uploaded_files = st.file_uploader("Sube tus reportes Excel", type=["xlsx"], accept_multiple_files=True)

    if uploaded_files:
        all_recopilacion = []
        all_descarga = []
        errores = []

        for uploaded_file in uploaded_files:
            try:
                st.write(f"Procesando archivo: {uploaded_file.name}")
                # Leer el archivo Excel
                wb = openpyxl.load_workbook(uploaded_file, data_only=True)
                sheets = wb.sheetnames
                st.write(f"Hojas encontradas: {sheets}")

                for sheet_name in sheets:
                    sheet = wb[sheet_name]
                    st.write(f"Procesando hoja: {sheet_name}")

                    # Extraer datos de celdas específicas
                    departamento = sheet["E6"].value or ""
                    provincia = sheet["L6"].value or ""
                    distrito = sheet["Q6"].value or ""
                    recopilador = sheet["Q8"].value or ""
                    lugar_de_desembarque = sheet["E8"].value or ""
                    fecha_cell = sheet["L8"].value

                    # Manejar casos donde Q8 y Q9 están combinados
                    if not recopilador:
                        recopilador = sheet["Q9"].value or ""

                    # Función para parsear la fecha
                    def parsear_fecha(fecha):
                        if isinstance(fecha, datetime):
                            return fecha.strftime("%d%m%Y")
                        elif isinstance(fecha, str):
                            try:
                                # Intentar parsear con dayfirst=True y fuzzy=True para mayor flexibilidad
                                fecha_dt = parser.parse(fecha, dayfirst=True, fuzzy=True)
                                return fecha_dt.strftime("%d%m%Y")
                            except parser.ParserError:
                                # Intentar reemplazar espacios por barras y volver a parsear
                                fecha_mod = fecha.replace(" ", "/")
                                try:
                                    fecha_dt = parser.parse(fecha_mod, dayfirst=True, fuzzy=True)
                                    return fecha_dt.strftime("%d%m%Y")
                                except parser.ParserError:
                                    return None
                        return None

                    # Parsear la fecha
                    fecha = parsear_fecha(fecha_cell)
                    if not fecha:
                        fecha = "01012024"  # Valor por defecto
                        errores.append(f"Fecha inválida en hoja {sheet_name} del archivo {uploaded_file.name}. Fecha original: '{fecha_cell}'")

                    # Extraer el lugar correctamente sin la fecha
                    lugar = extraer_lugar(lugar_de_desembarque)
                    st.write(f"Extraído 'lugar' como: {lugar}")

                    # Crear el campo 'registro'
                    registro = f"{lugar}{fecha}{sheet_name.replace(' ', '_')}"
                    st.write(f"Construido 'registro' como: {registro}")

                    # Obtener la observación
                    observacion = sheet["W13"].value or ""

                    # Preparar datos para 'recopilacion'
                    recopilacion_data = {
                        "departamento": departamento.strip() if isinstance(departamento, str) else departamento,
                        "provincia": provincia.strip() if isinstance(provincia, str) else provincia,
                        "distrito": distrito.strip() if isinstance(distrito, str) else distrito,
                        "recopilador": recopilador.strip() if isinstance(recopilador, str) else recopilador,
                        "registro": registro,
                        "observacion": observacion.strip() if isinstance(observacion, str) else ""
                    }

                    all_recopilacion.append(recopilacion_data)
                    st.write(f"Datos de recopilacion extraídos: {recopilacion_data}")

                    # Extraer datos de descarga
                    for row in range(13, 58):
                        nombre_cientifico = sheet[f"D{row}"].value or ""
                        cantidad = sheet[f"E{row}"].value
                        unidad_medida = sheet[f"F{row}"].value or ""
                        volumen = sheet[f"G{row}"].value
                        aparejo = sheet[f"H{row}"].value or ""
                        procedencia = sheet[f"I{row}"].value or ""
                        embarcacion = sheet[f"J{row}"].value or ""
                        matricula = sheet[f"K{row}"].value or ""
                        tripulantes = sheet[f"L{row}"].value
                        dias_de_faena = sheet[f"M{row}"].value
                        horas_de_faena = sheet[f"N{row}"].value
                        hora_de_descarga = sheet[f"O{row}"].value
                        tamano_1 = sheet[f"P{row}"].value
                        precio_tamano_1 = sheet[f"Q{row}"].value
                        tamano_2 = sheet[f"R{row}"].value
                        precio_tamano_2 = sheet[f"S{row}"].value
                        tamano_3 = sheet[f"T{row}"].value
                        precio_tamano_3 = sheet[f"U{row}"].value
                        destino = sheet[f"V{row}"].value or ""

                        # Standardizar tamano y precio
                        def estandarizar(valor):
                            if isinstance(valor, (int, float)):
                                return float(valor)
                            elif isinstance(valor, str):
                                try:
                                    # Reemplazar comas por puntos y convertir a float
                                    return float(valor.replace(",", "."))
                                except ValueError:
                                    return None
                            return None

                        tamano_1 = estandarizar(tamano_1)
                        tamano_2 = estandarizar(tamano_2)
                        tamano_3 = estandarizar(tamano_3)

                        precio_tamano_1 = estandarizar(precio_tamano_1)
                        precio_tamano_2 = estandarizar(precio_tamano_2)
                        precio_tamano_3 = estandarizar(precio_tamano_3)

                        # Función para parsear la hora
                        def parsear_hora(hora):
                            if isinstance(hora, datetime):
                                return hora.strftime("%H:%M")
                            elif isinstance(hora, time):
                                return hora.strftime("%H:%M")
                            elif isinstance(hora, str):
                                try:
                                    hora_dt = parser.parse(hora, fuzzy=True)
                                    return hora_dt.strftime("%H:%M")
                                except parser.ParserError:
                                    return None
                            return None

                        # Parsear hora de descarga
                        hora_descarga = parsear_hora(hora_de_descarga)
                        if not hora_descarga and hora_de_descarga:
                            errores.append(f"Hora de descarga inválida en fila {row} de hoja {sheet_name} del archivo {uploaded_file.name}. Hora original: '{hora_de_descarga}'")

                        descarga_data = {
                            "nombre_cientifico": nombre_cientifico.strip() if isinstance(nombre_cientifico, str) else nombre_cientifico,
                            "cantidad": cantidad if isinstance(cantidad, (int, float)) else None,
                            "unidad_medida": unidad_medida.strip() if isinstance(unidad_medida, str) else unidad_medida,
                            "volumen": volumen if isinstance(volumen, (int, float)) else None,
                            "aparejo": aparejo.strip() if isinstance(aparejo, str) else aparejo,
                            "procedencia": procedencia.strip() if isinstance(procedencia, str) else procedencia,
                            "embarcacion": embarcacion.strip() if isinstance(embarcacion, str) else embarcacion,
                            "matricula": matricula.strip() if isinstance(matricula, str) else matricula,
                            "tripulantes": tripulantes if isinstance(tripulantes, int) else None,
                            "dias_de_faena": dias_de_faena if isinstance(dias_de_faena, int) else None,
                            "horas_de_faena": horas_de_faena if isinstance(horas_de_faena, int) else None,
                            "hora_de_descarga": hora_descarga,
                            "tamano_1": tamano_1,
                            "precio_tamano_1": precio_tamano_1,
                            "tamano_2": tamano_2,
                            "precio_tamano_2": precio_tamano_2,
                            "tamano_3": tamano_3,
                            "precio_tamano_3": precio_tamano_3,
                            "destino": destino.strip() if isinstance(destino, str) else destino,
                            "registro": registro
                        }

                        # Validar si al menos 'nombre_cientifico' está presente
                        if nombre_cientifico:
                            all_descarga.append(descarga_data)
                            st.write(f"Datos de descarga extraídos fila {row}: {descarga_data}")

            except Exception as e:
                st.error(f"Error al procesar el archivo {uploaded_file.name}: {e}")
                errores.append(f"Error en archivo {uploaded_file.name}: {e}")

        # Mostrar errores si existen
        if errores:
            st.subheader("Errores Encontrados:")
            for error in errores:
                st.error(error)

        # Botón para enviar los datos a Supabase
        if st.button("Enviar a Supabase"):
            try:
                # Insertar en recopilacion
                if all_recopilacion:
                    recopilacion_df = pd.DataFrame(all_recopilacion).drop_duplicates(subset=["registro"])
                    records_recopilacion = recopilacion_df.to_dict(orient='records')
                    response_recopilacion = supabase.table("recopilacion").insert(records_recopilacion).execute()
                    
                    # Depuración: Verificar atributos del response
                    st.write("Respuesta de recopilacion:", response_recopilacion)
                    st.write("Atributos de response:", dir(response_recopilacion))
                    
                    # Usar 'status_code' para verificar el estado
                    if hasattr(response_recopilacion, 'status_code'):
                        if response_recopilacion.status_code in [200, 201]:
                            st.success("Todos los registros de recopilación fueron insertados correctamente.")
                        else:
                            # Acceder al mensaje de error si existe
                            error_message = response_recopilacion.error['message'] if response_recopilacion.error else "Error desconocido."
                            st.error(f"Error al insertar recopilación: {response_recopilacion.status_code} - {error_message}")
                    else:
                        st.error("La respuesta de Supabase para 'recopilacion' no tiene 'status_code'. Verifica la respuesta completa en las salidas de depuración.")

                # Insertar en descarga
                if all_descarga:
                    descarga_df = pd.DataFrame(all_descarga)
                    records_descarga = descarga_df.to_dict(orient='records')
                    response_descarga = supabase.table("descarga").insert(records_descarga).execute()
                    
                    # Depuración: Verificar atributos del response
                    st.write("Respuesta de descarga:", response_descarga)
                    st.write("Atributos de response:", dir(response_descarga))
                    
                    # Usar 'status_code' para verificar el estado
                    if hasattr(response_descarga, 'status_code'):
                        if response_descarga.status_code in [200, 201]:
                            st.success("Todos los registros de descarga fueron insertados correctamente.")
                        else:
                            # Acceder al mensaje de error si existe
                            error_message = response_descarga.error['message'] if response_descarga.error else "Error desconocido."
                            st.error(f"Error al insertar descarga: {response_descarga.status_code} - {error_message}")
                    else:
                        st.error("La respuesta de Supabase para 'descarga' no tiene 'status_code'. Verifica la respuesta completa en las salidas de depuración.")

                st.success("Todos los datos han sido enviados a Supabase correctamente.")

            except Exception as e:
                st.error(f"Error al enviar datos a Supabase: {e}")
