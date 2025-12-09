
# 📚 ProyectoIB – Sistema de Recuperación de Información

Este proyecto implementa un sistema **CLI (línea de comandos)** que permite realizar **consultas de texto libre** sobre el conjunto de documentos **ArguAna**.
Utiliza tres modelos clásicos de recuperación de información:

* 🔹 **Jaccard**
* 🔹 **TF-IDF**
* 🔹 **BM25**

Tras procesar la consulta, el sistema recupera y **ordena los documentos por relevancia**, de acuerdo con el modelo seleccionado por el usuario.

---

## ⚙️ Instalación

### 1️⃣ Clonar el repositorio

```bash
git clone https://github.com/ecazar/ProyectoIB.git
```

### 2️⃣ Instalar dependencias

Asegúrate de tener instalado Python 3.8+.

```bash
pip install -r requirements.txt
```

### 3️⃣ Descargar datos adicionales

```bash
python download_assets.py
```

---

## 🚀 Ejecución

### 🧪 Visualizar el código del sistema

Abrir el archivo Jupyter Notebook:

```
SistemaRecuperacion.ipynb
```

### 💻 Ejecutar la interfaz por consola

Desde la raíz del proyecto:

```bash
python interfaz_cli.py
```

---

## 🧩 Características principales

* 🔍 Búsquedas intuitivas en lenguaje natural
* 📊 Comparación entre modelos clásicos de IR
* ⚡ Respuesta rápida y ordenada por relevancia
* 🧹 Interfaz simple y fácil de usar

---

