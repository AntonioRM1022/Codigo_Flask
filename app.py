from flask import Flask, render_template, request, redirect, url_for, session, jsonify, send_file
from werkzeug.utils import secure_filename
from fpdf import FPDF  # fpdf2
from io import BytesIO
from PIL import Image, ExifTags
from datetime import datetime
from typing import cast
import os
import json
import time
import smtplib
import mimetypes
import numpy as np
import qrcode
import cv2

from pyzbar.pyzbar import decode
from email.message import EmailMessage
from ultralytics import YOLO
import openai
from openai import OpenAI



SKIP_EPP_VALIDATION =True # Cambia a False cuando quieras activar la validación

app = Flask(__name__)

model = YOLO('yolov_model/best.pt')
app.secret_key = 'clave_secreta_segura'
class PDF(FPDF):
    def header(self):
        self.set_font("DejaVu", style='BU', size=16)
        self.cell(0, 10, "Reporte mensual", new_x="LMARGIN", new_y="NEXT", align='C')

@app.route('/')
def home():
    return redirect(url_for('login'))

@app.route('/bienvenida')
def bienvenida():
    if 'usuario' not in session:
        return redirect(url_for('login'))
    return render_template('bienvenida.html')



@app.route('/login_qr')
def login_qr_page():
    return render_template('login_qr.html')
# Cargar usuarios desde archivo JSON
def cargar_usuarios():
    if os.path.exists("usuarios.json"):
        with open("usuarios.json", "r") as archivo:
            return json.load(archivo)
    return {}
@app.route('/admin')
def admin():
    if 'usuario' not in session or session.get('rol') != 'admin':
        return redirect(url_for('login'))

    with open('data/usuarios.json') as f:
        usuarios = json.load(f)

    return render_template('admin.html', usuarios=usuarios)

def insertar_imagen_corregida(pdf, user_id, x=20, y=30, box_size=50):
    foto_path = os.path.join('static', 'perfiles', f"{user_id}.jpg")
    if not os.path.exists(foto_path):
        return False

    try:
        img = Image.open(foto_path)

        # Corrige orientación
        for key, val in ExifTags.TAGS.items():
            if val == 'Orientation':
                orientation_key = key
                break
        exif = img._getexif()
        if exif and orientation_key in exif:
            orientation = exif[orientation_key]
            if orientation == 3:
                img = img.rotate(180, expand=True)
            elif orientation == 6:
                img = img.rotate(270, expand=True)
            elif orientation == 8:
                img = img.rotate(90, expand=True)

        # Redimensionar proporcionalmente para que quepa en 50x50 mm (≈ 190x190 px a 96 dpi)
        img.thumbnail((190, 190))

        # Guardar imagen temporal redimensionada
        temp_path = f"static/perfiles/temp_{user_id}.jpg"
        img.save(temp_path)

        # Calcular posición para centrar dentro del cuadro gris (50x50 mm)
        width_mm = img.width * 0.264583  # pixels to mm at 96 DPI
        height_mm = img.height * 0.264583
        offset_x = x + (box_size - width_mm) / 2
        offset_y = y + (box_size - height_mm) / 2

        # Dibuja cuadro gris
        pdf.set_fill_color(220, 220, 220)
        pdf.rect(x, y, box_size, box_size, style='F')

        # Insertar imagen centrada
        pdf.image(temp_path, x=offset_x, y=offset_y, w=width_mm)

        os.remove(temp_path)
        return True
    except Exception as e:
        print("Error procesando imagen:", e)
        return False


@app.route('/registrar_modal', methods=['POST','GET'])
def registrar_modal():
    try:
        userId = request.form.get('userId', '').strip()
        nombre = request.form.get('nombre', '').strip()
        clave = request.form.get('clave', '').strip()
        tarjeta = request.form.get('tarjeta', '').strip()
        rol = request.form.get('rol', '').strip()
        correo = request.form.get('correo', '').strip()
        archivo_imagen = request.files.get('imagen')

        if not all([userId, nombre, clave, tarjeta, rol, correo]):
            return jsonify(success=False, error="Todos los campos son obligatorios.")

        if not archivo_imagen or archivo_imagen.filename == '':
            return jsonify(success=False, error="Debes subir una imagen.")

        # Validar extensión
        extensiones_permitidas = {'png', 'jpg', 'jpeg', 'gif'}
        filename = secure_filename(archivo_imagen.filename)
        if '.' not in filename or filename.rsplit('.', 1)[1].lower() not in extensiones_permitidas:
            return jsonify(success=False, error="Formato de imagen no válido (png, jpg, jpeg, gif).")

        extension = filename.rsplit('.', 1)[1].lower()

        # Cargar usuarios
        ruta_usuarios = 'data/usuarios.json'
        usuarios = {}
        if os.path.exists(ruta_usuarios):
            with open(ruta_usuarios, 'r', encoding='utf-8') as f:
                usuarios = json.load(f)

        if userId in usuarios:
            return jsonify(success=False, error="El usuario ya existe.")

        # Guardar imagen
        nombre_guardado = f"{userId}.{extension}"
        ruta_perfiles = os.path.join('static', 'perfiles')
        os.makedirs(ruta_perfiles, exist_ok=True)
        ruta_imagen = os.path.join(ruta_perfiles, nombre_guardado)
        archivo_imagen.save(ruta_imagen)

        # Generar token y expiración
        token = f"token_seguro_{userId}_{int(time.time())}"
        expiracion = int(time.time()) + 30 * 24 * 3600  # 30 días

        usuarios[userId] = {
            "userId": userId,
            "nombre": nombre,
            "correo": correo,
            "clave": clave,
            "tarjeta": tarjeta,
            "rol": rol,
            "token": token,
            "expiracion": expiracion,
            "foto": nombre_guardado
        }

        with open(ruta_usuarios, 'w', encoding='utf-8') as f:
            json.dump(usuarios, f, indent=2, ensure_ascii=False)

        # Crear QR
        ruta_qr = os.path.join('static', 'qr', f"{userId}.png")
        os.makedirs(os.path.dirname(ruta_qr), exist_ok=True)
        qr_img = qrcode.make(userId)
        qr_img.save(ruta_qr)

        # Enviar QR al correo (opcional)
        try:
            enviar_qr_por_correo(correo, userId)
        except Exception as e:
            print(f"⚠️ Error al enviar correo: {e}")  # no bloquea el flujo

        return jsonify(success=True)

    except Exception as e:
        return jsonify(success=False, error=f"Error interno del servidor: {str(e)}")

@app.route('/admin/guardar_usuario/<usuario>', methods=['POST'])
def guardar_usuario(usuario):
    if 'usuario' not in session or session.get('rol') != 'admin':
        return redirect(url_for('login'))

    ruta_archivo = 'data/usuarios.json'
    with open(ruta_archivo, 'r', encoding='utf-8') as f:
        usuarios = json.load(f)

    if usuario not in usuarios:
        return "Usuario no encontrado", 404

    # Actualiza datos
    usuarios[usuario]['nombre'] = request.form.get('nombre', '')
    usuarios[usuario]['rol'] = request.form.get('rol', '')
    usuarios[usuario]['tarjeta'] = request.form.get('tarjeta', '')

    # Imagen
    if 'imagen' in request.files:
        imagen = request.files['imagen']
        if imagen and imagen.filename:
            ruta = os.path.join('static/perfiles', f"{usuario}.jpg")
            imagen.save(ruta)
            usuarios[usuario]['foto'] = f"{usuario}.jpg"

    # Guarda en archivo
    with open(ruta_archivo, 'w', encoding='utf-8') as f:
        json.dump(usuarios, f, indent=4, ensure_ascii=False)

    return redirect(url_for('admin'))


@app.route('/actualizar_usuario', methods=['POST'])
def actualizar_usuario():
    if 'usuario' not in session or session.get('rol') != 'admin':
        return redirect(url_for('login'))

    usuario = request.form.get('usuario')
    nombre = request.form.get('nombre')
    rol = request.form.get('rol')
    tarjeta = request.form.get('tarjeta')

    with open('data/usuarios.json', 'r', encoding='utf-8') as f:
        usuarios = json.load(f)

    if usuario not in usuarios:
        return jsonify(success=False, message="Usuario no encontrado"), 404

    # Actualizar campos
    usuarios[usuario]['nombre'] = nombre
    usuarios[usuario]['rol'] = rol
    usuarios[usuario]['tarjeta'] = tarjeta

    # Guardar imagen si se proporciona
    if 'imagen' in request.files:
        imagen = request.files['imagen']
        if imagen and imagen.filename:
            ruta = os.path.join('static/perfiles', f'{usuario}.jpg')
            imagen.save(ruta)
            usuarios[usuario]['foto'] = f'{usuario}.jpg'

    # Guardar cambios
    with open('data/usuarios.json', 'w', encoding='utf-8') as f:
        json.dump(usuarios, f, indent=2, ensure_ascii=False)

    return jsonify(success=True)
@app.route('/admin/eliminar_usuario', methods=['POST'])
def eliminar_usuario():
    if 'usuario' not in session or session.get('rol') != 'admin':
        return redirect(url_for('login'))

    data = request.get_json()
    usuario = data.get('usuario')

    if not usuario:
        return jsonify(success=False, error="Usuario no especificado")

    ruta_usuarios = 'data/usuarios.json'
    if not os.path.exists(ruta_usuarios):
        return jsonify(success=False, error="Archivo de usuarios no encontrado")

    with open(ruta_usuarios, 'r', encoding='utf-8') as f:
        usuarios = json.load(f)

    if usuario not in usuarios:
        return jsonify(success=False, error="Usuario no encontrado")

    # Eliminar la imagen si existe
    nombre_imagen = usuarios[usuario].get('foto')
    if nombre_imagen:
        ruta_imagen = os.path.join('static', 'perfiles', nombre_imagen)
        if os.path.exists(ruta_imagen):
            os.remove(ruta_imagen)

    # Eliminar el código QR si existe
    ruta_qr = os.path.join('static', 'qr', f'{usuario}.png')
    if os.path.exists(ruta_qr):
        os.remove(ruta_qr)

    # Eliminar el usuario del JSON
    usuarios.pop(usuario)

    with open(ruta_usuarios, 'w', encoding='utf-8') as f:
        json.dump(usuarios, f, indent=2, ensure_ascii=False)

    return jsonify(success=True)



@app.route('/supervisor')
def supervisor():
    if 'usuario' not in session or session.get('rol') != 'supervisor':
        return redirect(url_for('login'))

    with open('data/usuarios.json', encoding='utf-8') as f:
        usuarios = json.load(f)

        empleados = [
            {
                "nombre": usuario.get("nombre", user_id),
                "usuario": user_id,
                "tarjeta": usuario.get("tarjeta", ""),
                "rol": usuario.get("rol", ""),
                "imagen": f"perfiles/{user_id}.jpg"
            }
            for user_id, usuario in usuarios.items()
            if usuario.get("rol") == "empleado"
    ]

    return render_template('supervisor.html', personal=empleados)

@app.route('/reportes')
def reportes():
    if 'usuario' not in session or session.get('rol') != 'supervisor':
        return redirect(url_for('login'))
    # Aquí deberías cargar y pasar los reportes reales
    reportes_lista = [
        {"id": 1, "titulo": "Reporte de Incidencias", "fecha": "2025-06-20"},
        {"id": 2, "titulo": "Reporte de Mantenimiento", "fecha": "2025-06-21"},
    ]
    return render_template('reportes.html', reportes=reportes_lista)


@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))
@app.route('/empleado')


@app.route('/empleado')
def vista_empleado():
    if 'usuario' not in session or session.get('rol') != 'empleado':
        return redirect(url_for('login'))

    userId = session['usuario']
    tarjeta = session.get('tarjeta', '')

    # Cargar datos del usuario
    with open('data/usuarios.json', 'r', encoding='utf-8') as f:
        usuarios = json.load(f)

    usuario_data = usuarios.get(userId, {})
    nombre = usuario_data.get('nombre', 'Usuario desconocido')
    rol = usuario_data.get('rol', 'sin rol')

    # Leer equipos desde JSON
    with open('data/equipos.json', 'r', encoding='utf-8') as f:
        equipos = json.load(f)

    # Leer archivos PDF desde la carpeta 'static/manuals'
    manuals_dir = os.path.join(app.static_folder, 'manuals')
    manuales = []
    if os.path.exists(manuals_dir):
        manuales = [f for f in os.listdir(manuals_dir) if f.endswith('.pdf')]

    return render_template('empleado.html',
                           userId=userId,
                           nombre=nombre,
                           rol=rol,
                           tarjeta=tarjeta,
                           equipos=equipos,
                           manuales=manuales)

# Guardar usuarios en archivo JSON
def guardar_usuarios(usuarios):
    with open("usuarios.json", "w") as archivo:
        json.dump(usuarios, archivo, indent=4)

def guardar_usuario_json(userId, nombre, clave, tarjeta, rol):
    ruta_archivo = os.path.join("data", "usuarios.json")

    token = f"token_seguro_{userId}_{int(time.time())}"
    expiracion = int(time.time()) + 60*60*24*30

    nuevo_usuario = {
        userId: {
            "userId": userId,
            "nombre": nombre,
            "clave": clave,
            "tarjeta": tarjeta,
            "rol": rol,
            "token": token,
            "expiracion": expiracion
        }
    }

    os.makedirs("data", exist_ok=True)

    try:
        with open(ruta_archivo, "r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        data = {}

    data.update(nuevo_usuario)

    with open(ruta_archivo, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    return nuevo_usuario[userId]
from werkzeug.utils import secure_filename

@app.route('/registro', methods=['GET', 'POST'])
def registro():
    if request.method == 'POST':
        userId = request.form['userId']
        nombre = request.form['nombre']
        clave = request.form['clave']
        tarjeta = request.form['tarjeta']
        rol = request.form['rol']
        correo = request.form['correo']

        # Obtener archivo de imagen
        archivo_imagen = request.files.get('imagen')

        if not archivo_imagen:
            return render_template('registro.html', error="⚠️ Debes subir una foto de perfil")

        # Validar extensión (opcional)
        extensiones_permitidas = {'png', 'jpg', 'jpeg', 'gif'}
        filename = secure_filename(archivo_imagen.filename)
        if '.' in filename:
            extension = filename.rsplit('.', 1)[1].lower()
            if extension not in extensiones_permitidas:
                return render_template('registro.html', error="⚠️ Tipo de archivo no permitido")
        else:
            return render_template('registro.html', error="⚠️ Archivo sin extensión")

        # Cargar usuarios existentes
        ruta_usuarios = 'data/usuarios.json'
        if os.path.exists(ruta_usuarios):
            with open(ruta_usuarios, 'r', encoding='utf-8') as f:
                usuarios = json.load(f)
        else:
            usuarios = {}

        if userId in usuarios:
            return render_template('registro.html', error="⚠️ El usuario ya existe")

        # Guardar imagen en static/perfiles con nombre userId + extensión
        nombre_guardado = f"{userId}.{extension}"
        ruta_perfil = os.path.join('static', 'perfiles')
        os.makedirs(ruta_perfil, exist_ok=True)
        ruta_imagen = os.path.join(ruta_perfil, nombre_guardado)
        archivo_imagen.save(ruta_imagen)

        # Crear token y expiración
        token = f"token_seguro_{userId}_{int(time.time())}"
        expiracion = int(time.time()) + 30 * 24 * 3600

        usuarios[userId] = {
            "userId": userId,
            "nombre": nombre,
            "correo": correo,
            "clave": clave,
            "tarjeta": tarjeta,
            "rol": rol,
            "token": token,
            "expiracion": expiracion,
            "foto": nombre_guardado  # Guardamos el nombre de archivo para mostrar después
        }

        # Guardar usuarios
        with open(ruta_usuarios, 'w', encoding='utf-8') as f:
            json.dump(usuarios, f, indent=2, ensure_ascii=False)

        # Generar QR
        ruta_qr = f"static/qr/{userId}.png"
        os.makedirs(os.path.dirname(ruta_qr), exist_ok=True)
        img = qrcode.make(userId)
        img.save(ruta_qr)

        # Enviar correo
        enviar_qr_por_correo(correo, userId)

        return render_template('registro.html', mostrar_modal=True, nuevo_usuario=usuarios[userId])

    return render_template('registro.html')



def enviar_qr_por_correo(destinatario, user_id):
    remitente = "antonioreach102@gmail.com"
    contraseña = "rbxg jjza rnjr djoj"  # Usar clave generada de App en Gmail

    msg = EmailMessage()
    msg['Subject'] = f'QR de acceso - {user_id}'
    msg['From'] = remitente
    msg['To'] = destinatario
    msg.set_content(f'Aquí está el QR de acceso para el usuario: {user_id}')

    qr_path = f"static/qr/{user_id}.png"
    tipo_mime, _ = mimetypes.guess_type(qr_path)
    tipo_principal, tipo_sub = tipo_mime.split('/')

    with open(qr_path, 'rb') as f:
        msg.add_attachment(f.read(), maintype=tipo_principal, subtype=tipo_sub, filename=f"{user_id}.png")

    with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
        smtp.login(remitente, contraseña)
        smtp.send_message(msg)



@app.route('/captura')
def captura():
    cam = cv2.VideoCapture(0)
    ret, frame = cam.read()
    if ret:
        ruta_img = 'static/img/captura.jpg'
        cv2.imwrite(ruta_img, frame)
        cam.release()

        resultados = model.predict(ruta_img)[0]
        objetos_detectados = []
        for r in resultados.boxes:
            clase = model.names[int(r.cls)]
            objetos_detectados.append(clase)

        with open('data/equipos.json') as f:
            db = json.load(f)

        equipo_detectado = next((eq for eq in db if eq in objetos_detectados), None)

        if equipo_detectado:
            return redirect(url_for('manual', nombre=equipo_detectado))
        else:
            return "No se detectó equipo reconocido."
    return "Error al capturar imagen."
@app.route('/manual/<nombre>')
def manual(nombre):
    if 'usuario' not in session or 'tarjeta' not in session:
        return redirect(url_for('login'))

    try:
        with open('data/equipos.json', 'r', encoding='utf-8') as f:
            equipos = json.load(f)
    except FileNotFoundError:
        return "<h3>⚠️ No se encontró el archivo de equipos.</h3><a href='/empleado'>Volver</a>"

    equipo = equipos.get(nombre)
    if not equipo:
        return f"<h3>⚠️ Equipo no encontrado: {nombre}</h3><a href='/empleado'>Volver</a>"

    # Validar tarjeta del usuario
    tarjeta_usuario = session['tarjeta']
    tarjeta_requerida = equipo.get('tarjeta_requerida')
    # Mapeo de nivel
    niveles = {'A': 1, 'B': 2, 'C': 3}
    nivel_usuario = niveles.get(tarjeta_usuario, 0)
    nivel_requerido = niveles.get(tarjeta_requerida, 3)

    if nivel_usuario < nivel_requerido:
        return f"""
        <h3>🔒 Acceso restringido</h3>
        <p>Este equipo requiere tarjeta tipo <strong>{tarjeta_requerida}</strong>.</p>
        <a href='/empleado'>🔙 Volver</a>
        """

    # Ruta del PDF del manual
    manual_url = url_for('static', filename='manuals/' + equipo['manual'])
    nombre_legible = nombre.replace('_', ' ').capitalize()

    return render_template('manual.html', nombre=nombre_legible, equipo=equipo, manual_url=manual_url)



@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        usuario = request.form['usuario']
        clave = request.form['clave']

        with open('data/usuarios.json') as f:
            usuarios = json.load(f)

        if usuario in usuarios and usuarios[usuario]['clave'] == clave:
            session['usuario'] = usuario
            session['tarjeta'] = usuarios[usuario]['tarjeta']
            session['rol'] = usuarios[usuario]['rol']
            return redirect(url_for('bienvenida'))  # Redirige al splash de bienvenida
        else:
            return render_template('login.html', error="Credenciales incorrectas")

    return render_template('login.html')


@app.route('/login_qr_captura', methods=['GET'])
def login_qr_captura():
    cam = cv2.VideoCapture(0)
    captura_programada = False
    tiempo_inicio = None
    segundos_espera = 3
    frame_capturado = None

    while True:
        ret, frame = cam.read()
        if not ret:
            cam.release()
            return "<h3>No se pudo acceder a la cámara</h3><a href='/login'>Volver</a>"

        # Decodificar QR
        codigos = decode(frame)

        if codigos and not captura_programada:
            # Detectó QR, iniciar temporizador
            print("QR detectado, se tomará captura en 3 segundos...")
            captura_programada = True
            tiempo_inicio = cv2.getTickCount() / cv2.getTickFrequency()

        if captura_programada:
            tiempo_actual = cv2.getTickCount() / cv2.getTickFrequency()
            tiempo_transcurrido = tiempo_actual - tiempo_inicio
            tiempo_restante = int(segundos_espera - tiempo_transcurrido) + 1

            # Mostrar cuenta regresiva en la imagen
            cv2.putText(frame, f"Captura en {tiempo_restante} s", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            if tiempo_transcurrido >= segundos_espera:
                frame_capturado = frame.copy()
                break  # Salir del loop para procesar captura

        # Mostrar el frame con texto si aplica
        cv2.imshow("Escaneo QR", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cam.release()
            cv2.destroyAllWindows()
            return redirect(url_for('login'))

    cam.release()
    cv2.destroyAllWindows()

    if frame_capturado is None:
        return "<h3>Error al capturar la imagen</h3><a href='/login'>Volver</a>"

    # Guardar la imagen capturada
    ruta_img = 'static/img/qr_login.jpg'
    cv2.imwrite(ruta_img, frame_capturado)

    # Leer la imagen y decodificar QR
    imagen = cv2.imread(ruta_img)
    codigos = decode(imagen)

    if not codigos:
        return "<h3>⚠️ No se detectó ningún QR</h3><a href='/login'>Volver</a>"

    data_qr = codigos[0].data.decode('utf-8')

    with open('data/usuarios.json') as f:
        usuarios = json.load(f)

    if data_qr in usuarios:
        session['usuario'] = data_qr
        session['tarjeta'] = usuarios[data_qr]['tarjeta']
        session['rol'] = usuarios[data_qr]['rol']
        return redirect(url_for('bienvenida'))  # REDIRIGE AL SPLASH
    else:
        return "<h3>⚠️ Usuario QR no válido</h3><a href='/login'>Volver</a>"


@app.route('/login_qr_validar', methods=['POST'])
def login_qr_validar():
    data = request.get_json()
    qr_data = data.get("qr")

    with open('data/usuarios.json') as f:
        usuarios = json.load(f)

    if qr_data in usuarios:
        session['usuario'] = qr_data
        session['tarjeta'] = usuarios[qr_data]['tarjeta']
        session['rol'] = usuarios[qr_data]['rol']
        return jsonify({"success": True, "redirect": url_for('bienvenida')})  # REDIRIGE AL SPLASH
    else:
        return jsonify({"success": False, "message": "⚠️ Usuario QR no válido"})
@app.route('/verificar_epp')
@app.route('/verificar_epp_web')
def verificar_epp():
    if 'usuario' not in session or 'rol' not in session or 'tarjeta' not in session:
            return redirect(url_for('login'))

    if SKIP_EPP_VALIDATION:
            rol = session['rol']
            if rol == 'admin':
                return redirect(url_for('admin'))
            elif rol == 'supervisor':
                return redirect(url_for('supervisor'))
            elif rol == 'empleado':
                return redirect(url_for('vista_empleado'))
            else:
                return "<h3>⚠️ Rol desconocido</h3><a href='/login'>Volver</a>"

    tarjeta = session['tarjeta']
    rol = session['rol']
    requerimientos = {
        "A": ["casco", "chaleco"],
        "B": ["casco", "chaleco", "guantes"],
        "C": ["casco", "chaleco", "guantes", "arnes"]
    }

    requeridos = requerimientos.get(tarjeta, [])

    return render_template("verificar_epp.html", requeridos=requeridos)

@app.route('/api/verificar_epp', methods=['POST'])
def api_verificar_epp():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400

    file = request.files['image']
    npimg = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    # Corremos la detección
    results = model(img)[0]  # tomamos el primer resultado (batch=1)

    detecciones = []
    etiquetas = []

    for r in results.boxes.data.tolist():
        # r contiene [x1, y1, x2, y2, score, class_id]
        x1, y1, x2, y2, score, class_id = r
        label = model.names[int(class_id)]

        # Filtrar detecciones por confianza si quieres (ej: >0.5)
        if score < 0.5:
            continue

        etiquetas.append(label)
        detecciones.append({
            'label': label,
            'x1': int(x1),
            'y1': int(y1),
            'x2': int(x2),
            'y2': int(y2)
        })

    # Quitamos etiquetas duplicadas
    etiquetas = list(set(etiquetas))

    return jsonify({'etiquetas': etiquetas, 'detecciones': detecciones})
@app.route('/escaner', methods=['GET', 'POST'])
def escaner():
    if 'usuario' not in session or 'tarjeta' not in session:
        return redirect(url_for('login'))

    if request.method == 'GET':
        return render_template('escaner.html')

    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400

    file = request.files['image']
    img_bytes = file.read()

    # Convertir a imagen OpenCV
    npimg = np.frombuffer(img_bytes, np.uint8)
    frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    # Detectar QR
    codigos_qr = decode(frame)
    if not codigos_qr:
        return jsonify({'acceso': None})  # No se detectó QR

    equipo_id = codigos_qr[0].data.decode('utf-8')

    # Leer equipos
    with open('data/equipos.json', encoding='utf-8') as f:
        equipos = json.load(f)

    equipo = equipos.get(equipo_id)
    if not equipo:
        return jsonify({'acceso': None})  # QR no válido

    tarjeta_usuario = session['tarjeta']
    tarjeta_requerida = equipo.get('tarjeta_requerida')

    if tarjeta_usuario >= tarjeta_requerida:
        return jsonify({'acceso': True, 'equipo': equipo_id})
    else:
        # Obtener lista de personas con tarjeta suficiente
        with open('data/usuarios.json', encoding='utf-8') as f:
            usuarios = json.load(f)

        responsables = [
            {'nombre': nombre, 'rol': datos['rol']}
            for nombre, datos in usuarios.items()
            if datos.get('tarjeta', 'Z') <= tarjeta_usuario and datos.get('rol') in ['supervisor', 'admin']
        ]

        return jsonify({
            'acceso': False,
            'equipo': equipo_id,
            'responsables': responsables
        })
@app.route('/equipos')
def lista_equipos():
    if 'rol' not in session:
        return redirect(url_for('login'))

    # Solo empleados y superiores pueden ver
    if session['rol'] not in ['admin', 'supervisor', 'empleado']:
        return "Acceso denegado", 403

    return render_template('equipos.html', equipos=equipos, herramientas=herramientas, rol=session['rol'])
@app.route('/equipos/nuevo', methods=['GET', 'POST'])
def crear_equipo():
    if 'rol' not in session or session['rol'] != 'admin':
        return "Acceso denegado", 403

    if request.method == 'POST':
        nuevo_id = max(e['id'] for e in equipos) + 1 if equipos else 1
        nombre = request.form['nombre']
        equipos.append({"id": nuevo_id, "nombre": nombre, "herramientas": []})
        return redirect(url_for('lista_equipos'))

    return render_template('crear_equipo.html')
@app.route('/forgot_password', methods=['GET', 'POST'])
def forgot_password():
    if request.method == 'POST':
        user_id = request.form['userId']
        nueva = request.form['nueva']
        confirmar = request.form['confirmar']

        if nueva != confirmar:
            return render_template('forgot_password.html', error="⚠️ Las contraseñas no coinciden.")

        # Cargar base de usuarios
        with open('data/usuarios.json', 'r', encoding='utf-8') as f:
            usuarios = json.load(f)

        if user_id in usuarios:
            usuarios[user_id]['clave'] = nueva
            with open('data/usuarios.json', 'w', encoding='utf-8') as f:
                json.dump(usuarios, f, indent=2, ensure_ascii=False)
            return redirect(url_for('login'))
        else:
            return render_template('forgot_password.html', error="⚠️ Usuario no encontrado.")

    return render_template('forgot_password.html')
@app.route('/registrar_qr', methods=['POST'])
def registrar_qr():
    data = request.get_json()
    tipo = data.get('tipo')         # 'qraprobado' o 'qrprohibido'
    qr = data.get('qr')
    userId = data.get('userId')

    ruta_usuarios = os.path.join('data', 'usuarios.json')
    with open(ruta_usuarios, 'r', encoding='utf-8') as f:
        usuarios = json.load(f)

    if userId in usuarios:
        usuario = usuarios[userId]

        # ⬆️ Contador
        usuario[tipo] = usuario.get(tipo, 0) + 1

        # 📄 Lista de manuales vistos
        usuario.setdefault("manuales_vistos", []).append(qr)

        # Guardar cambios
        with open(ruta_usuarios, 'w', encoding='utf-8') as f:
            json.dump(usuarios, f, indent=2, ensure_ascii=False)

        return jsonify({"mensaje": "registro actualizado"}), 200

    return jsonify({"error": "usuario no encontrado"}), 404

@app.route('/admin/editar_usuario/<usuario>', methods=['POST'])
def editar_usuario(usuario):
    try:
        datos_nuevos = request.get_json()
        nombre = datos_nuevos.get('nombre')
        rol = datos_nuevos.get('rol')
        tarjeta = datos_nuevos.get('tarjeta')

        ruta_usuarios = 'data/usuarios.json'
        if not os.path.exists(ruta_usuarios):
            return jsonify(success=False, error="Base de datos no encontrada"), 404

        with open(ruta_usuarios, 'r', encoding='utf-8') as f:
            usuarios = json.load(f)

        if usuario not in usuarios:
            return jsonify(success=False, error="Usuario no encontrado"), 404

        usuarios[usuario]['nombre'] = nombre
        usuarios[usuario]['rol'] = rol
        usuarios[usuario]['tarjeta'] = tarjeta

        with open(ruta_usuarios, 'w', encoding='utf-8') as f:
            json.dump(usuarios, f, indent=2, ensure_ascii=False)

        return jsonify(success=True)

    except Exception as e:
        return jsonify(success=False, error=f"Error al editar: {str(e)}"), 500

@app.route('/validar_equipo/<codigo>', methods=['POST'])
def validar_equipo(codigo):
    data = request.get_json()
    userId = data.get("userId")

    if not userId:
        return jsonify({"acceso": False, "motivo": "usuario no proporcionado"}), 400

    # Cargar datos
    with open('data/usuarios.json', encoding='utf-8') as f:
        usuarios = json.load(f)
    with open('data/equipos.json', encoding='utf-8') as f:
        equipos = json.load(f)

    # Validar usuario
    if userId not in usuarios:
        return jsonify({"acceso": False, "motivo": "usuario inválido"}), 403

    usuario = usuarios[userId]
    tarjeta_usuario = usuario.get("tarjeta")

    # Validar equipo
    equipo = equipos.get(codigo)
    if not equipo:
        return jsonify({"acceso": False, "motivo": "equipo no existe"}), 404

    tarjeta_requerida = equipo.get("tarjeta_requerida")

    if tarjeta_usuario == tarjeta_requerida:
        return jsonify({"acceso": True, "nombre_equipo": codigo})

    return jsonify({
        "acceso": False,
        "motivo": "Tarjeta no autorizada para este equipo."
    }), 401


@app.route("/empleado_info/<user_id>")
def empleado_info(user_id):
    with open("data/usuarios.json", encoding="utf-8") as f:
        usuarios = json.load(f)

    if user_id not in usuarios:
        return jsonify({"error": "Usuario no encontrado"}), 404

    user = usuarios[user_id]
    return jsonify({
        "qraprobado": user.get("qraprobado", 0),
        "qrprohibido": user.get("qrprohibido", 0)
    })





@app.route('/reporte_pdf/<userId>')
def generar_reporte_pdf(userId):
    with open('data/usuarios.json', encoding='utf-8') as f:
        usuarios = json.load(f)

    if userId not in usuarios:
        return "Usuario no encontrado", 404

    usuario = usuarios[userId]
    nombre = usuario.get('nombre', 'Desconocido')
    rol = usuario.get('rol', 'Sin rol')
    aprobados = usuario.get('qraprobado', 0)
    denegados = usuario.get('qrprohibido', 0)
    manuales = usuario.get('manuales_vistos', [])

    # Generar leyenda IA
    prompt = f"""
    Genera una leyenda profesional para el reporte mensual de {nombre}, con rol de {rol}.
    Ha escaneado {aprobados} códigos QR válidos y {denegados} denegados.
    Manuales vistos: {', '.join(manuales) if manuales else 'ninguno'}.
    """
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7
    )
    leyenda_ia = response.choices[0].message.content.strip()

    # Limitar texto IA
    if len(leyenda_ia) > 700:
        leyenda_ia = leyenda_ia[:700] + "..."

    # Crear PDF
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", style='B', size=16)
    pdf.set_text_color(0, 51, 102)
    pdf.cell(0, 10, "Reporte mensual", ln=True, align='C')
    pdf.ln(5)

    # Imagen
    insertar_imagen_corregida(pdf, userId, x=20, y=30, box_size=50)

    # Texto principal
    pdf.set_xy(80, 30)
    pdf.set_text_color(0, 0, 0)
    pdf.set_font("Arial", size=12)
    pdf.cell(0, 10, f"Nombre: {nombre}", ln=True)
    pdf.cell(0, 10, f"Rol: {rol}", ln=True)

    mes_actual = datetime.now().strftime("%B %Y")
    pdf.cell(0, 10, f"Historial de escaneos ({mes_actual}):", ln=True)
    pdf.cell(0, 10, f"QR Aprobados: {aprobados}", ln=True)
    pdf.cell(0, 10, f"QR Denegados: {denegados}", ln=True)


    pdf.ln(5)
    pdf.set_font("Arial", style='B', size=12)
    pdf.cell(0, 10, "Manuales escaneados:", ln=True)
    pdf.set_font("Arial", size=11)

    if manuales:
        for i, m in enumerate(manuales, 1):
            pdf.cell(0, 8, f"{i}. {m}", ln=True)
    else:
        pdf.cell(0, 8, "No ha escaneado manuales aún.", ln=True)

    pdf.ln(10)
    pdf.set_font("Arial", size=11)
    pdf.multi_cell(0, 8, leyenda_ia)

    # Guardar PDF en carpeta
    fecha_actual = datetime.now().strftime("%Y-%m-%d")
    nombre_archivo = f"reporte_{userId}_{fecha_actual}.pdf"
    ruta_guardado = os.path.join("reportes", nombre_archivo)
    os.makedirs("reportes", exist_ok=True)
    pdf.output(ruta_guardado)

    # Enviar PDF al navegador
    buffer = BytesIO()
    pdf.output(buffer)
    buffer.seek(0)

    return send_file(
        buffer,
        as_attachment=False,
        download_name=nombre_archivo,
        mimetype="application/pdf"
    )
@app.route('/equipos/<int:id>/editar', methods=['GET', 'POST'])
def editar_equipo(id):
    if 'rol' not in session or session['rol'] != 'admin':
        return "Acceso denegado", 403

    equipo = next((e for e in equipos if e['id'] == id), None)
    if not equipo:
        return "Equipo no encontrado", 404

    if request.method == 'POST':
        equipo['nombre'] = request.form['nombre']
        return redirect(url_for('lista_equipos'))

    return render_template('editar_equipo.html', equipo=equipo)

@app.route('/equipos/<int:id>/borrar', methods=['POST'])
def borrar_equipo(id):
    if 'rol' not in session or session['rol'] != 'admin':
        return "Acceso denegado", 403

    global equipos
    equipos = [e for e in equipos if e['id'] != id]
    return redirect(url_for('lista_equipos'))
@app.route('/herramientas')
def lista_herramientas():
    if 'rol' not in session:
        return redirect(url_for('login'))

    if session['rol'] not in ['admin', 'supervisor', 'empleado']:
        return "Acceso denegado", 403

    return render_template('herramientas.html', herramientas=herramientas, rol=session['rol'])
@app.route('/herramientas/nuevo', methods=['GET', 'POST'])
def crear_herramienta():
    if 'rol' not in session or session['rol'] != 'admin':
        return "Acceso denegado", 403

    if request.method == 'POST':
        nuevo_id = max(h['id'] for h in herramientas) + 1 if herramientas else 1
        nombre = request.form['nombre']
        herramientas.append({"id": nuevo_id, "nombre": nombre})
        return redirect(url_for('lista_herramientas'))

    return render_template('crear_herramienta.html')

@app.route('/herramientas/<int:id>/editar', methods=['GET', 'POST'])
def editar_herramienta(id):
    if 'rol' not in session or session['rol'] != 'admin':
        return "Acceso denegado", 403

    herramienta = next((h for h in herramientas if h['id'] == id), None)
    if not herramienta:
        return "Herramienta no encontrada", 404

    if request.method == 'POST':
        herramienta['nombre'] = request.form['nombre']
        return redirect(url_for('lista_herramientas'))

    return render_template('editar_herramienta.html', herramienta=herramienta)
@app.route('/herramientas/<int:id>/borrar', methods=['POST'])
def borrar_herramienta(id):
    if 'rol' not in session or session['rol'] != 'admin':
        return "Acceso denegado", 403

    global herramientas
    herramientas = [h for h in herramientas if h['id'] != id]

    # También eliminar de equipos que la tengan asignada
    for equipo in equipos:
        if id in equipo['herramientas']:
            equipo['herramientas'].remove(id)

    return redirect(url_for('lista_herramientas'))
@app.route('/equipos/<int:id>/asignar', methods=['GET', 'POST'])
def asignar_herramientas(id):
    if 'rol' not in session or session['rol'] not in ['admin', 'supervisor']:
        return "Acceso denegado", 403

    equipo = next((e for e in equipos if e['id'] == id), None)
    if not equipo:
        return "Equipo no encontrado", 404

    if request.method == 'POST':
        herramientas_seleccionadas = request.form.getlist('herramientas')
        equipo['herramientas'] = list(map(int, herramientas_seleccionadas))
        return redirect(url_for('lista_equipos'))

    return render_template('asignar_herramientas.html', equipo=equipo, herramientas=herramientas)
@app.route('/cambiar_imagen/<usuario>', methods=['GET', 'POST'])
@app.route('/cambiar_imagen_modal', methods=['POST'])
def cambiar_imagen_modal():
    usuario = request.form['usuario']
    archivo = request.files.get('imagen')
    ruta_usuarios = 'data/usuarios.json'

    if not os.path.exists(ruta_usuarios):
        return "No hay usuarios registrados", 404

    with open(ruta_usuarios, 'r', encoding='utf-8') as f:
        usuarios = json.load(f)

    if usuario not in usuarios:
        return "Usuario no encontrado", 404

    if archivo and archivo.filename != '':
        extension = os.path.splitext(archivo.filename)[1]
        nombre_archivo = f"{usuario}{extension}"
        ruta_imagen = os.path.join('static/perfiles', nombre_archivo)
        os.makedirs('static/perfiles', exist_ok=True)
        archivo.save(ruta_imagen)

        # Guardar ruta relativa solo el nombre del archivo en el campo "foto"
        usuarios[usuario]['foto'] = nombre_archivo

        with open(ruta_usuarios, 'w', encoding='utf-8') as f:
            json.dump(usuarios, f, indent=2, ensure_ascii=False)

        return '', 200

    return 'Archivo inválido', 400


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)