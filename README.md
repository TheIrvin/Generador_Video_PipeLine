# Generador Video Pipeline

API local para animar una imagen fija con un MP4 opcional como guía de movimiento.

El motor no genera contenido con difusión. Si el primer fotograma de `motion_reference` coincide con la imagen enviada, conserva los fotogramas de esa animación y los adapta al FPS y resolución solicitados; así no pierde las transformaciones y efectos que no se pueden reconstruir desde una sola imagen. Si el guía muestra otra escena, estima la cámara y transfiere movimiento con flujo óptico. Sin guía, crea solo el movimiento de cámara configurado. Ninguno de estos modos sintetiza partes ocultas ni acciones nuevas.

## Requisitos

- Python 3.10 o posterior.
- Dependencias Python de `requirements.txt`.
- No hace falta instalar FFmpeg aparte: `imageio-ffmpeg` incluye el binario que usa el servicio.
- Para el preset del ejemplo: 10 s, 24 fps, 720×1280 (240 frames).

## Instalar y ejecutar

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
uvicorn animavideo.main:app --app-dir src --host 127.0.0.1 --port 8000
```

La documentación interactiva de la API queda en `http://127.0.0.1:8000/docs`; `GET /health` comprueba el servicio.

También se puede ejecutar como contenedor con `Copy-Item .env.example .env` y `docker compose up --build -d`. Cambia antes `VIDEO_PIPELINE_API_KEY` en `.env`; configura `VIDEO_PIPELINE_CORS_ORIGINS` si habrá llamadas desde un navegador. El volumen `./data/jobs` conserva trabajos y resultados.

## API para conectar otro pipeline

Crear un trabajo con `POST /v1/jobs` como `multipart/form-data`:

- `image`: archivo JPEG, PNG o WebP.
- `prompt`: prompt original, conservado en los metadatos para trazabilidad.
- `options`: objeto JSON opcional con parámetros del render.
- `motion_reference`: MP4 opcional de referencia. Si empieza con la misma imagen subida, sus fotogramas son la referencia visual de la animación; en caso contrario, solo se transfiere el movimiento estimado.

Ejemplo:

```powershell
curl.exe -X POST http://127.0.0.1:8000/v1/jobs `
  -F "image=@C:\ruta\referencia.jpg" `
  -F "motion_reference=@C:\ruta\clip_guia.mp4" `
  -F "prompt=Anima únicamente la imagen de referencia; haz un acercamiento lento y anima el vórtice visible." `
  -F 'options={"duration_seconds":10,"fps":24,"width":720,"height":1280,"preset":"motion_transfer"}'
```

La respuesta incluye `job_id`, `status_url` y `video_url`. Consulta `GET /v1/jobs/{job_id}` hasta que `status` sea `completed`, y después descarga el MP4 con `GET /v1/jobs/{job_id}/video`. Los estados posibles son `queued`, `processing`, `completed` y `failed`.

Parámetros disponibles: `duration_seconds`, `fps`, `width`, `height`, `preset`, `zoom_start`, `zoom_end`, `focal_x`, `focal_y` y `hold_seconds`. Para movimiento transferido, los fotogramas del MP4 guía se adaptan a la resolución/FPS de salida. Por omisión se produce un MP4 de 10 s, 24 FPS y 720×1280.

La carpeta de trabajos por omisión es `data/jobs`; se puede cambiar con `VIDEO_PIPELINE_DATA_DIR`. Cada trabajo conserva su imagen, `job.json` y el MP4 de salida. El directorio está excluido de Git para evitar publicar los archivos del usuario.

Al iniciar, el servicio marca como `failed` los trabajos que quedaron en `queued` o `processing` por una interrupción. Los trabajos terminados conservan su estado y resultados; los trabajos interrumpidos incluyen un mensaje que permite distinguirlos de un error del render.

Los renders se procesan de uno en uno. Para integraciones desde un navegador, configura `VIDEO_PIPELINE_CORS_ORIGINS` con los orígenes permitidos separados por coma. Si el servicio será accesible fuera del equipo local, define `VIDEO_PIPELINE_API_KEY`; el cliente debe enviarla como `X-API-Key`. El servidor arranca enlazado a `127.0.0.1`; expón otro host solo detrás de una red y reglas de acceso que controles.

## Pruebas

Instala las dependencias de desarrollo y ejecuta las pruebas sin iniciar un render real:

```powershell
python -m pip install -e ".[test]"
python -m pytest
ruff check src tests
```

GitHub Actions ejecuta esas comprobaciones en Python 3.10, 3.12 y 3.14.

## n8n

n8n es opcional. Otro flujo puede llamar la API con un nodo HTTP Request, consultar el estado y descargar el resultado. Mantener el motor como servicio HTTP evita acoplarlo al shell de n8n y permite llamarlo desde otros sistemas. Si n8n corre en Docker, debe poder alcanzar el host/API y las rutas compartidas solo son necesarias si el cliente envía archivos por ruta en vez de subirlos al endpoint.

## Prompt ajustado

El prompt original y el prompt ajustado quedan en `job.json` para trazabilidad. La plantilla está en `prompts/animation-motion-transfer.txt`; los parámetros de `options` controlan el render y el texto no invoca un modelo generativo.

## Licencia

Este proyecto está bajo la licencia MIT. Consulta [LICENSE](LICENSE).
