# Configuración de RunPod Serverless desde Cero

Este repositorio contiene la plantilla base para desplegar el worker de inferencia de IA para **Aura Style AI** en RunPod Serverless.

El worker se encarga de:
1. Descargar la imagen selfie del usuario desde el bucket temporal de S3 (`s3://...`).
2. Procesar la simulación capilar (con InstantID / SDXL).
3. Guardar el resultado procesado directamente en la ruta de salida de S3 (`s3://...`).

---

## Paso 1: Construir y Publicar la Imagen Docker

El worker debe empaquetarse en una imagen Docker y subirse a un registro público (ej. Docker Hub, GitHub Container Registry) para que RunPod pueda descargarla.

1. **Autentícate en tu registro Docker**:
   ```bash
   docker login
   ```

2. **Construye la imagen Docker**:
   ```bash
   docker build -t <tu-usuario-docker>/aurastyle-runpod-worker:latest .
   ```

3. **Sube la imagen a Docker Hub**:
   ```bash
   docker push <tu-usuario-docker>/aurastyle-runpod-worker:latest
   ```

---

## Paso 2: Configurar las Credenciales de AWS S3 en RunPod

Para que el worker de RunPod pueda leer y escribir en tus buckets S3 temporales, necesita credenciales válidas de AWS.

1. Entra a tu consola de **[RunPod Console](https://www.runpod.io/)**.
2. Ve a la sección **Environment Variables** en el panel lateral (o agrégalas directamente a la plantilla en el siguiente paso).
3. Recomendamos configurar las siguientes variables de entorno para tu plantilla/endpoint:
   * `AWS_ACCESS_KEY_ID`: Tu access key de AWS.
   * `AWS_SECRET_ACCESS_KEY`: Tu secret key de AWS.
   * `AWS_DEFAULT_REGION`: La región donde se encuentra tu infraestructura (ej. `us-east-1`).

---

## Paso 3: Crear la Plantilla (Template) en RunPod

1. Ve a **Templates** en la consola de RunPod.
2. Haz clic en **New Template**.
3. Completa el formulario con la siguiente información:
   * **Template Name**: `aurastyle-hair-simulation`
   * **Container Image**: `<tu-usuario-docker>/aurastyle-runpod-worker:latest` (o el URI de tu imagen Docker).
   * **Container Disk**: `10 GB` (o lo requerido por el modelo).
   * **Volume Disk**: `0 GB` (los workers de Serverless recomiendan usar disco de contenedor en vez de volumen persistente para reducir cold starts).
4. En **Environment Variables**, añade las variables de AWS del **Paso 2**:
   * Key: `AWS_ACCESS_KEY_ID` | Value: `[Tu Key]`
   * Key: `AWS_SECRET_ACCESS_KEY` | Value: `[Tu Secret]`
   * Key: `AWS_DEFAULT_REGION` | Value: `[Tu Región]`
5. Haz clic en **Save Template**.

---

## Paso 4: Crear el Endpoint Serverless en RunPod

1. Ve a **Serverless** -> **Endpoints** en la barra lateral.
2. Haz clic en **New Endpoint**.
3. Configura las siguientes opciones:
   * **Endpoint Name**: `aurastyle-simulation-endpoint`
   * **Select Template**: Selecciona `aurastyle-hair-simulation` (creada en el Paso 3).
   * **Min Workers**: `0` (se apaga cuando no hay tráfico para no consumir saldo, produciendo cold start).
   * **Max Workers**: `3` (límite máximo de escalabilidad).
   * **Idle Timeout**: `60` segundos (tiempo para apagar el contenedor después de la última petición).
   * **Select GPU Type(s)**: Selecciona `NVIDIA RTX 4090` o `NVIDIA RTX 3090` (24 GB de VRAM son idóneos para InstantID).
4. Haz clic en **Create**.
5. Una vez creado, verás una pantalla con la información de tu endpoint. **Copia el ID del Endpoint** (ej. `y4m3g0h5q2w8v1`). Este valor será tu `RunPodEndpointId`.

---

## Paso 5: Obtener la API Key de RunPod

1. Ve a **Settings** -> **API Keys** en la consola de RunPod.
2. Haz clic en **Add API Key**.
3. Asígnale un nombre (ej. `Aura Style Dev Key`) y haz clic en **Create**.
4. **Copia y guarda la API Key generada** en un lugar seguro. No podrás volver a verla. Este valor será tu `RunPodApiKey`.

---

## Paso 6: Vincular RunPod con el Entorno de Desarrollo Local

Ahora que tienes la **API Key** y el **Endpoint ID**, configura SST para que la infraestructura en vivo los use al orquestar las peticiones.

1. Abre tu terminal en el directorio `/home/st-t/aurastyle-app`.
2. Si vas a levantar el entorno de desarrollo local con `npx sst dev`, vincula las credenciales ejecutando:
   ```bash
   npx sst secret set RunPodApiKey "<TU_API_KEY>"
   npx sst secret set RunPodEndpointId "<TU_ENDPOINT_ID>"
   ```
3. Esto actualizará el almacén de secretos de AWS SSM y los inyectará automáticamente en las lambdas locales y remotas.
