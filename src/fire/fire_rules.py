import numpy as np

from .board import Board, CellState, WallState, Direction


def roll_target(board: Board, rng: np.random.Generator) -> tuple[int, int]:
    row = int(rng.integers(0, board.height))
    col = int(rng.integers(0, board.width))
    return (row, col)


def advance_fire(board: Board, rng: np.random.Generator) -> None:
    pos = roll_target(board, rng)
    state = board.get_state(pos)

    match state:
        case CellState.CLEAR:
            board.set_state(pos, CellState.SMOKE)
        case CellState.SMOKE:
            board.set_state(pos, CellState.FIRE)
        case CellState.FIRE:
            explosion(board, pos)
        case _:
            pass

    flashover(board, pos)


def explosion(board: Board, pos: tuple[int, int]) -> None:
    for direction in Direction:
        resolve_shockwave(board, pos, direction)


def resolve_shockwave(board: Board, pos: tuple[int, int], direction: Direction) -> None:
    current = pos

    while True:
        wall = board.get_wall(current, direction)
        if wall is None:
            return

        if wall == WallState.WALL:
            board.set_wall(current, direction, WallState.DAMAGED_WALL)
            return
        if wall == WallState.DAMAGED_WALL:
            board.set_wall(current, direction, WallState.CLEAR)
            return
        if wall == WallState.DOOR_CLOSE:
            board.set_wall(current, direction, WallState.CLEAR)
            return

        nxt = board.neighbor(current, direction)
        if nxt is None:
            return

        state = board.get_state(nxt)
        if state == CellState.FIRE:
            current = nxt
            continue
        if state == CellState.SMOKE:
            board.set_state(nxt, CellState.FIRE)
            return
        # NOTE: victim and poi dissappear here, counter needs to be set
        board.set_state(nxt, CellState.FIRE)
        return


def flashover(board: Board, pos: tuple[int, int]) -> None:
    stack = [pos]
    visited = {pos}
    while stack:
        current = stack.pop()
        state = board.get_state(current)
        if state not in (CellState.SMOKE, CellState.FIRE):
            continue
        if state == CellState.SMOKE:
            board.set_state(current, CellState.FIRE)

        for direction in Direction:
            if not board.is_passable(current, direction):
                continue
            nxt = board.neighbor(current, direction)
            if nxt is not None and nxt not in visited:
                visited.add(nxt)
                stack.append(nxt)
