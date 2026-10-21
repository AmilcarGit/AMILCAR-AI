# AMILCAR AI — AMI Nano 2.0

AMI Nano 2.0 es un experimento de red neuronal de caracteres que se entrena desde cero y ejecuta localmente con Python estándar. No utiliza APIs, pesos preentrenados, PyTorch ni NumPy. Está pensado como MVP educativo para Termux/Android ARM64, no como sustituto de un LLM moderno.

## Instalar y probar en Termux

```bash
pkg install nodejs python git -y
git clone -b feat/ami-nano-termux https://github.com/AmilcarGit/AMILCAR-AI.git
cd AMILCAR-AI
npm install
npm run ami:train
npm run ami:chat -- "hola"
npm run ami:test
```

Si ya tienes la rama clonada, ejecuta `git pull origin feat/ami-nano-termux` antes de probar. La instalación no descarga modelos de internet. El entrenamiento genera `artifacts/ami-nano.json` localmente.

## ¿Qué arquitectura tiene?

Una red neuronal mínima: el carácter anterior se representa como índice de entrada y una matriz de pesos entrenables produce una distribución softmax del siguiente carácter. Los pesos empiezan aleatoriamente y se actualizan con descenso de gradiente usando pares de caracteres del corpus. Es un modelo de contexto de un carácter, pequeño y limitado; no es un Transformer ni entiende semántica.

- `data/corpus.txt`: ejemplos de entrenamiento editables.
- `ami-nano/train.py`: entrenamiento y guardado atómico del checkpoint.
- `ami-nano/generate.py`: inferencia local.
- `tests-ami/test_ami_nano.py`: pruebas de entrenamiento, generación y errores.
- `AMI_SEED`: semilla de entrenamiento (por defecto 7).
- `AMI_EPOCHS`: épocas (por defecto 80; máximo 2000).
- `AMI_MAX_NEW_TOKENS`: longitud máxima de salida (por defecto 180; máximo 500).
- `AMI_TEMPERATURE`: diversidad de generación (por defecto 0.85).

## Límites y seguridad

La calidad depende directamente del corpus; un modelo tan pequeño producirá errores y texto poco coherente. No entrenes con mensajes privados ni secretos. Usa solo textos propios o con licencia compatible. El bot no modifica automáticamente el código ni se autoactualiza en producción.

## Siguientes fases

1. Evaluar calidad con un conjunto de preguntas reservadas que no se incluyan en el entrenamiento.
2. Ampliar el contexto y comparar métricas reproducibles.
3. Añadir una interfaz estable del SDK para BaseBot sin cambiar los demás comandos.
4. Considerar un motor C/C++ únicamente después de perfilar el rendimiento.
