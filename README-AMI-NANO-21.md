# AMI Nano Evolutiva 2.1

Motor conversacional experimental para español, completamente local y sin APIs externas.

## Qué hace
- Recupera respuestas revisadas desde `data/dialogos-es.json`.
- Incluye conocimientos explicativos locales iniciales para fotosíntesis, aprendizaje e inteligencia artificial.
- Compone respuestas con plantillas y explica su origen.
- Solo presenta texto neuronal como propuesta experimental si supera controles léxicos básicos; estos controles no prueban que sea verdadero.
- Si no encuentra información fiable, reconoce la limitación en lugar de inventar.

## Probar en Termux ARM64
```sh
npm run ami:test
npm run ami:chat-hibrido -- "¿Quién eres?"
npm run ami:chat-hibrido -- "Explícame la fotosíntesis de las plantas"
npm run ami:chat-hibrido -- "¿Qué opinas de aprender cosas nuevas?"
```

No necesita instalar paquetes Python externos. El entrenamiento neuronal existente sigue siendo un modelo de caracteres pequeño, no un modelo de lenguaje avanzado. La cobertura depende de los conocimientos revisados incluidos en el repositorio.

## Aprendizaje y privacidad
Las conversaciones no se publican ni se añaden automáticamente al conocimiento público. Los ejemplos nuevos deben revisarse antes de incorporarlos a la base compartida.
