from .model import FirefighterModel


def run_simulation(csv_path: str | None = None, simulation_id: int = 0) -> FirefighterModel:
    """Corre una partida completa (desde cero hasta ganar/perder) sobre
    el tablero inicial fijo de la simulación, imprime un resumen del
    resultado, y devuelve el modelo ya terminado por si se quiere
    inspeccionar más. Si se pasa `csv_path`, exporta el dataset completo
    (cada turno: celdas, paredes, bomberos, POIs, acciones, fuego) a ese
    archivo CSV."""
    model = FirefighterModel()
    model.collector.simulation_id = simulation_id
    status = model.run()
    print(f"Simulación terminada: {status}")
    print(f"  Turnos jugados: {model.turn}")
    print(f"  Víctimas rescatadas: {model.victims_rescued}")
    print(f"  Víctimas perdidas: {model.victims_lost}")
    print(f"  Daño total: {model.damage_total}")
    if csv_path:
        model.collector.to_csv(csv_path)
        print(f"  Dataset exportado a: {csv_path}")
    return model

