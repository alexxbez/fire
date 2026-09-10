# Flashpoint Simulación

Simulación basada en el juego Flash Point: Fire Rescue. Construido con [Mesa](https://mesa.readthedocs.io/) y [Flask](https://flask.palletsprojects.com/).

Para la materia TC2008B, el reto final.

## Requisitos

- Python 3.14+

Se recomienda utilizar `uv` para evitar problemas de versiones y dependencias.

- [uv](https://docs.astral.sh/uv/)

## Instalación

```bash
git clone && cd fire
uv sync
```

## Ejecución

### Server de simulación

```bash
uv run src/fire/server.py
```

### Correr experimentos (N simulaciones por tipo de agente)

```bash
uv run scripts/run_experiments.py --sims 1000
```

### Análisis de resultados

```bash
uv run scripts/analyze_results.py
```

Los gráficos se guardan en `data/plots/`.

## Estructura del proyecto

```
src/
  fire/
    model.py          - Modelo principal (FirefighterModel)
    agents.py         - Agentes (estrategia mejorada)
    random_agent.py   - Agente con comportamiento aleatorio
    board.py          - Tablero y reglas del entorno
    fire_rules.py     - Lógica de propagación del fuego
    data_collector.py - Recolección de datos por turno
    server.py         - Servidor Flask para visualización
scripts/
  run_experiments.py  - Correr simulaciones en lote
  analyze_results.py  - Generar reportes y gráficos
data/                 - CSVs de resultados y gráficos generados
```

## Agentes

- **Smart**: Asigna víctimas por proximidad, prioriza supresión de fuego y rescate de forma coordinada.
- **Random**: Toma decisiones aleatorias (utilizado como línea base comparativa).

## Condiciones de victoria/derrota

- **Victoria**: rescatar >= 7 víctimas
- **Derrota**: perder >= 4 víctimas **o** daño total >= 24 (colapso del edificio)

## Uso de IA

Algunas partes de este proyecto fueron generadas con ayuda de la IA. A continuación se mencionan estas partes.

Sin embargo, cabe recalcar que las estrategias, algoritmos, y lógica principal no fue generada con IA.

**General**

- La documentación de las funciones, métodos y clases fue generada con IA.
- Las convenciones de nombres, así como el `_` para métodos privados, fue sugerido por la IA.

**`model.py`**

- La sugerencia de un argumento `agent_cls` fue hecha por IA.
- Bug en `_assign_victims` resuelto con ayuda de IA.
- Bug en `_assign_fire_targets` resuelto con ayuda de IA. Multiplicador por pared también agregado con ayuda de IA.

**`agents.py`**

- Bug en `_weighed_graph` identificado por IA. La solución de la IA fue usada con ligeros cambios.
- Sugerencia de permitir varios `targets` en `_weighed_path_to`.
- Bug en `move` para evitar dos poi en la misma casilla.
- Error de python en `take_turn`.
- Bug en `_extinguish_..` corregido por IA.
- Cálculo de `approach` dentro de `_fire_suppression_strategy` asistido por IA.

**`random_agent.py`**

- Reutilización de la clase `Firefighter` sugerido e implementado por la IA.
- Funciones auxiliares implementadas por IA.

**`fire_rules.py`**

- Sugerencia de clase `FireEvent`.
- Bug en `flashover` y `smoke_flashover` identificado por IA.

**`board.py`**

- Implementación de atributo `claim` en `POI`.
- Sugerencia de utilizar `Direction` con los valores en tupla.
- Ayuda con documentación de métodos de `Networkx`.
- Implementación de las posiciones hardcodeadas dentro de `create_board`.

**`data_collector.py`**

- La estructura del json fue diseñada por nosotros. Sin embargo, la construcción del objeto y creación de este fue programado por IA.

**`run_experiments.py`**

- Este archivo fue mayormente creado por IA. Manualmente ejectua las simulaciones en lugar de utilizar un batch runner. Loggea los resultados. Guarda los resultados en un csv para analizar posteriormente.

**`analyze_results.py`**

- Este archivo fue mayormente creado por IA. Toma los csvs generados y realiza un análisis, creando gráficas también.

**`README.md`**

- La IA se utilizó como ayuda para redactar este README.
