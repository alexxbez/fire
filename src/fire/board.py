import networkx as nx
from enum import Enum, auto
import numpy as np

BOARD_HEIGHT = 8
BOARD_WIDTH = 10


class CellState(Enum):
    CLEAR     = auto()
    SMOKE     = auto()
    FIRE      = auto()

class WallState(Enum):
    CLEAR        = auto()
    DOOR_OPEN    = auto()
    DOOR_CLOSE   = auto()
    WALL         = auto()
    DAMAGED_WALL = auto()

class PoiState(Enum):
    VICTIM  = auto()
    UNKNOWN = auto()
    FALSE   = auto()

class Direction(Enum):
    UP    = (-1, 0)
    DOWN  = (1, 0)
    LEFT  = (0, -1)
    RIGHT = (0, 1)

    @property
    def opposite(self):
        return {
            Direction.UP: Direction.DOWN,
            Direction.DOWN: Direction.UP,
            Direction.LEFT: Direction.RIGHT,
            Direction.RIGHT: Direction.LEFT,
        }[self]


class Board:
    def __init__(self, height: int = BOARD_HEIGHT, width: int = BOARD_WIDTH, pois = []):
        self.height = height
        self.width = width
        self.pois = pois
        self.G = nx.grid_2d_graph(height, width)

        nx.set_node_attributes(
            self.G, {node: CellState.CLEAR for node in self.G.nodes}, "state"
        )
        nx.set_node_attributes(
            self.G, {node: True for node in self.G.nodes}, "is_empty"
        )
        nx.set_edge_attributes(
            self.G, {edge: WallState.CLEAR for edge in self.G.edges}, "wall"
        )

    # ---- cells ----------------------------------------------------

    def in_bounds(self, pos: tuple[int, int]) -> bool:
        row, col = pos
        return 0 <= row < self.height and 0 <= col < self.width

    def get_state(self, pos: tuple[int, int]) -> CellState:
        return self.G.nodes[pos]["state"]

    def set_state(self, pos: tuple[int, int], state: CellState) -> None:
        self.G.nodes[pos]["state"] = state

    def is_empty(self, pos: tuple[int, int]) -> bool:
        return self.G.nodes[pos]["is_empty"]

    def clear_cell(self, pos: tuple[int, int]) -> None:
        self.G.nodes[pos]["is_empty"] = True

    def fill_cell(self, pos: tuple[int, int]) -> None:
        self.G.nodes[pos]["is_empty"] = False

    def is_outside(self, pos: tuple[int, int]) -> bool:
        row, col = pos
        return row == 0 or row == self.height - 1 \
            or col == 0 or col == self.width - 1        

    # ---- walls / neighbors -----------------------------------------

    def neighbor(self, pos: tuple[int, int], direction: Direction) -> tuple[int, int] | None:
        row, col = pos
        drow, dcol = direction.value
        n = (row + drow, col + dcol)
        return n if self.in_bounds(n) else None

    def get_wall(self, pos: tuple[int, int], direction: Direction) -> WallState | None:
        n = self.neighbor(pos, direction)
        if n is None:
            return None
        return self.G[pos][n]["wall"]

    def set_wall(self, pos: tuple[int, int], direction: Direction, state: WallState) -> None:
        n = self.neighbor(pos, direction)
        if n is None:
            return
        self.G[pos][n]["wall"] = state

    def is_passable(self, pos: tuple[int, int], direction: Direction) -> bool:
        """Does not account for other agents"""
        wall = self.get_wall(pos, direction)
        if wall is None:
            return False
        return wall in (WallState.CLEAR, WallState.DOOR_OPEN)

    def open_neighbors(self, pos: tuple[int, int]) -> list[tuple[int, int]]:
        result = []
        for direction in Direction:
            if self.is_passable(pos, direction):
                n = self.neighbor(pos, direction)
                if n is not None and self.is_empty(n):
                    result.append(n)
        return result

    # ---- pois  --------------------------------------------------------

    def has_poi(self, pos: tuple[int, int]) -> bool:
        return bool(list(filter(lambda x: x[0] == pos, self.pois)))

    def reveal_poi(self, pos: tuple[int, int], rng: np.random.Generator) -> PoiState:
        choices = [PoiState.VICTIM, PoiState.FALSE]
        choice = choices[rng.integers(0, len(choices))]
        if choice == PoiState.FALSE:
            self.pois = list(filter(lambda x: x[0] != pos, self.pois))
        else:
            self.pois = list(map(lambda x: x if x[0] != pos else (pos, choice), self.pois))
        return choice

    def move_victim(self, pos: tuple[int, int], target: tuple[int, int]) -> None:
        self.pois = list(map(lambda x: x if x[0] != pos else (target, x[1]), self.pois))

    # ---- debugging ----------------------------------------------------

    def dump(self) -> None:
        for node, data in self.G.nodes(data=True):
            print(node, data["state"])


def create_board() -> Board:
    board = Board(pois=[((2, 4), PoiState.UNKNOWN), ((5, 1), PoiState.UNKNOWN), ((5, 8), PoiState.UNKNOWN)])

    # --- fires -------------------------------------------------------
    fires = [
        (2, 2), (2, 3), (3, 2), (3, 3), (3, 4),
        (3, 5), (4, 4), (5, 6), (5, 7), (6, 6),
    ]
    for pos in fires:
        board.set_state(pos, CellState.FIRE)

    # --- outer walls --------------------------------------------------
    for row in range(board.height):
        board.set_wall((row, 0), Direction.LEFT, WallState.WALL)
        board.set_wall((row, board.width - 1), Direction.RIGHT, WallState.WALL)
    for col in range(board.width):
        board.set_wall((0, col), Direction.UP, WallState.WALL)
        board.set_wall((board.height - 1, col), Direction.DOWN, WallState.WALL)

    # --- interior walls -----------------------------------------------
    interior_walls = [
        ((2, 3), Direction.RIGHT),
        ((1, 5), Direction.DOWN),
        ((4, 2), Direction.RIGHT),
        ((3, 6), Direction.RIGHT),
        ((5, 5), Direction.RIGHT),
        ((5, 7), Direction.RIGHT),
    ]
    for pos, direction in interior_walls:
        board.set_wall(pos, direction, WallState.WALL)

    # --- doors ---------------------------------------------------------
    doors = [
        ((1, 3), Direction.DOWN, WallState.DOOR_CLOSE),
        ((0, 6), Direction.DOWN, WallState.DOOR_OPEN),
        ((2, 5), Direction.RIGHT, WallState.DOOR_CLOSE),
        ((3, 0), Direction.UP, WallState.DOOR_OPEN),
        ((3, 2), Direction.RIGHT, WallState.DOOR_CLOSE),
        ((2, 8), Direction.DOWN, WallState.DOOR_CLOSE),
        ((4, 6), Direction.DOWN, WallState.DOOR_CLOSE),
        ((4, 8), Direction.RIGHT, WallState.DOOR_OPEN),
        ((4, 4), Direction.DOWN, WallState.DOOR_CLOSE),
        ((6, 5), Direction.DOWN, WallState.DOOR_CLOSE),
        ((6, 7), Direction.DOWN, WallState.DOOR_CLOSE),
        ((6, 3), Direction.DOWN, WallState.DOOR_OPEN),
    ]
    for pos, direction, state in doors:
        board.set_wall(pos, direction, state)

    return board


if __name__ == "__main__":
    b = create_board()
    b.dump()
