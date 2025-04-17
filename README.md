# CIUT Tablero

Un tablero interactivo para visualizar el riesgo de inundación en La Plata, Berisso y Ensenada.

## Descripción

Esta aplicación permite visualizar rápidamente capas relacionadas con el riesgo de inundación en La Plata y sus alrededores. Es un producto mínimo viable (MVP) que utiliza:

- **Streamlit** para la interfaz de usuario y el despliegue
- **geemap** para interactuar con Google Earth Engine
- **Google Earth Engine** como fuente de datos geoespaciales
- **uv** para la gestión de paquetes
- **ruff** para el linting y formateo de código

Este proyecto está ligeramente basado en [el trabajo de Qiusheng Wu](https://github.com/opengeos/streamlit-geospatial).

## Instalación

1. Clona este repositorio:

```bash
git clone https://github.com/nlebovits/ciut-tablero.git
cd ciut-tablero
```

2. Instala las dependencias usando uv:

```bash
uv sync
```

Si necesitas actualizar el archivo de bloqueo:

```bash
uv lock
```

3. Configura las credenciales de Google Earth Engine:

Crea una carpeta `.streamlit` en el directorio raíz y dentro de ella un archivo `secrets.toml` con el siguiente contenido:

```toml
GOOGLE_SERVICE_ACCOUNT_KEY =
    """
    TU_CLAVE_CODIFICADA_EN_BASE64
    """
GOOGLE_CLOUD_PROJECT_NAME = "tu-proyecto-gee"
```

> **Nota**: Necesitarás registrar un proyecto en Google Cloud Platform para usar Earth Engine y descargar la clave de servicio correspondiente. Deberás convertir la clave de servicio de JSON a formato base64.

## Uso

Para ejecutar la aplicación localmente:

```bash
uv run streamlit run streamlit_app.py
```

## Despliegue

Para desplegar en Streamlit Cloud:

1. Sube tu repositorio a GitHub
2. Conéctalo a Streamlit Cloud
3. Añade las mismas variables secretas (GOOGLE_SERVICE_ACCOUNT_KEY y GOOGLE_CLOUD_PROJECT_NAME) en la configuración de secretos de tu aplicación en Streamlit Cloud

## Licencia

Este proyecto está licenciado bajo [GNU General Public License](LICENSE).
