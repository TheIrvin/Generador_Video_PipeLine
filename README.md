# Generador Video Pipeline

API local para transformar una imagen fija en un clip vertical con movimiento de cámara y animación ligera de efectos ya visibles. El primer preset replica el ejemplo anime: acercamiento al personaje y vórtice energético magenta/dorado.

El motor no regenera al personaje con un modelo de difusión. Aplica transformaciones y composición sobre la imagen de entrada, así que conserva la ilustración y produce un movimiento controlado. Los cambios de pose, acciones nuevas y segmentación automática avanzada quedan fuera de esta primera versión.

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

Ejemplo:

```powershell
curl.exe -X POST http://127.0.0.1:8000/v1/jobs `
  -F "image=@C:\ruta\referencia.jpg" `
  -F "prompt=Anima únicamente la imagen de referencia; haz un acercamiento lento y anima el vórtice visible." `
  -F 'options={"duration_seconds":10,"fps":24,"width":720,"height":1280,"preset":"energy_push_in"}'
```

La respuesta incluye `job_id`, `status_url` y `video_url`. Consulta `GET /v1/jobs/{job_id}` hasta que `status` sea `completed`, y después descarga el MP4 con `GET /v1/jobs/{job_id}/video`. Los estados posibles son `queued`, `processing`, `completed` y `failed`.

Parámetros disponibles: `duration_seconds`, `fps`, `width`, `height`, `preset`, `zoom_start`, `zoom_end`, `focal_x`, `focal_y`, `hold_seconds`, `animate_energy`, `energy_x`, `energy_y`, `energy_radius`, `energy_turns`, `energy_growth` y `energy_strength`. Las coordenadas focales y del efecto son relativas al ancho/alto, en el rango 0–1. Los valores por omisión están ajustados al JPG del ejemplo.

La carpeta de trabajos por omisión es `data/jobs`; se puede cambiar con `VIDEO_PIPELINE_DATA_DIR`. Cada trabajo conserva su imagen, `job.json` y el MP4 de salida. El directorio está excluido de Git para evitar publicar los archivos del usuario.

Los renders se procesan de uno en uno. Para integraciones desde un navegador, configura `VIDEO_PIPELINE_CORS_ORIGINS` con los orígenes permitidos separados por coma. Si el servicio será accesible fuera del equipo local, define `VIDEO_PIPELINE_API_KEY`; el cliente debe enviarla como `X-API-Key`. El servidor arranca enlazado a `127.0.0.1`; expón otro host solo detrás de una red y reglas de acceso que controles.

## n8n

n8n es opcional. Otro flujo puede llamar la API con un nodo HTTP Request, consultar el estado y descargar el resultado. Mantener el motor como servicio HTTP evita acoplarlo al shell de n8n y permite llamarlo desde otros sistemas. Si n8n corre en Docker, debe poder alcanzar el host/API y las rutas compartidas solo son necesarias si el cliente envía archivos por ruta en vez de subirlos al endpoint.

## Prompt ajustado

El prompt de animación usado como intención de este preset está en `prompts/animation-energy-push-in.txt`. Los parámetros de `options` controlan el render; el texto no invoca un modelo generativo.
