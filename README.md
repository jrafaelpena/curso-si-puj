## Setup de entorno virtual con uv

### UV 

[uv](https://docs.astral.sh/uv/): An extremely fast Python package and project manager, written in Rust.

Link para instalación en Linux o Windows: [Link](https://docs.astral.sh/uv/getting-started/installation/#__tabbed_1_2)

Instalación de python usando uv (si no está la versión instalada en el sistema)

```Bash
uv python install 3.12.11
```

Creación de entorno virtual **en la carpeta del proyecto**

```Bash
# uv sync --group=<grupo-dependencias-modulo-específico>
uv sync --group=rn
```
