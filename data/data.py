import pandas as pd
import sqlalchemy
from sqlalchemy import create_engine
import os

def generar_sql_create_e_insert(excel_file, nombre_tabla, archivo_sql_salida):
    """
    Lee un archivo Excel y genera sentencias SQL CREATE TABLE e INSERT INTO basadas en las columnas y datos.

    :param excel_file: Ruta al archivo Excel (por ejemplo, 'data.xlsx')
    :param nombre_tabla: Nombre de la tabla que se creará en SQL
    :param archivo_sql_salida: Ruta al archivo donde se guardará el código SQL (por ejemplo, 'script.sql')
    """
    # Leer el archivo Excel
    df = pd.read_excel(excel_file)

    # Mapeo de tipos de datos de pandas a SQL
    def map_dtype(dtype):
        if pd.api.types.is_integer_dtype(dtype):
            return "INT"
        elif pd.api.types.is_float_dtype(dtype):
            return "FLOAT"
        elif pd.api.types.is_bool_dtype(dtype):
            return "BOOLEAN"
        elif pd.api.types.is_datetime64_any_dtype(dtype):
            return "DATETIME"
        else:
            # Asumir que es una cadena de texto
            # Puedes ajustar el tamaño según tus necesidades
            return "VARCHAR(255)"

    # Generar definiciones de columnas
    columnas_sql = []
    columnas_originales = df.columns.tolist()
    columnas_formateadas = []
    for columna in columnas_originales:
        tipo_sql = map_dtype(df[columna].dtype)
        # Reemplazar espacios y caracteres especiales en nombres de columnas
        columna_sql = columna.replace('°', '').replace(' ', '_').replace('-', '_')
        columnas_sql.append(f"  `{columna_sql}` {tipo_sql}")
        columnas_formateadas.append(columna_sql)

    # Unir todas las definiciones de columnas para CREATE TABLE
    columnas_definicion = ",\n".join(columnas_sql)
    sql_create = f"CREATE TABLE `{nombre_tabla}` (\n{columnas_definicion}\n);\n\n"

    # Generar sentencias INSERT INTO
    # Preparar las columnas para el INSERT
    columnas_insert = ", ".join([f"`{col}`" for col in columnas_formateadas])

    # Inicializar las sentencias INSERT
    sql_insert = f"INSERT INTO `{nombre_tabla}` ({columnas_insert}) VALUES\n"

    # Función para escapar y formatear valores
    def formato_valor(valor):
        if pd.isnull(valor):
            return "NULL"
        elif isinstance(valor, str):
            # Escapar comillas simples en cadenas
            valor_escapado = valor.replace("'", "''")
            return f"'{valor_escapado}'"
        elif isinstance(valor, (int, float)):
            return str(valor)
        elif isinstance(valor, pd.Timestamp):
            return f"'{valor.strftime('%Y-%m-%d %H:%M:%S')}'"
        else:
            # Para otros tipos, convertir a cadena y escapar
            valor_escapado = str(valor).replace("'", "''")
            return f"'{valor_escapado}'"

    # Generar los valores para cada fila
    valores_filas = []
    for index, row in df.iterrows():
        valores = ", ".join([formato_valor(row[col]) for col in columnas_originales])
        valores_filas.append(f"({valores})")

        # Opcional: Limitar la cantidad de filas por INSERT para evitar scripts demasiado grandes
        if (index + 1) % 1000 == 0:
            sql_insert += ",\n".join(valores_filas) + ";\n\n"
            valores_filas = []

    # Agregar los últimos valores
    if valores_filas:
        sql_insert += ",\n".join(valores_filas) + ";\n"

    # Combinar CREATE TABLE e INSERT INTO
    sql_completo = sql_create + sql_insert

    # Guardar el SQL en un archivo
    with open(archivo_sql_salida, 'w', encoding='utf-8') as archivo_sql:
        archivo_sql.write(sql_completo)

    print(f"El script SQL se ha guardado en '{archivo_sql_salida}'.")

# Ejemplo de uso
if __name__ == "__main__":
    excel_path = 'data.xlsx'             # Ruta a tu archivo Excel
    tabla_sql = 'data'               # Nombre deseado para la tabla en SQL
    archivo_sql = 'data.sql' # Nombre del archivo SQL de salida

    generar_sql_create_e_insert(excel_path, tabla_sql, archivo_sql)
