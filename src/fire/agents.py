from mesa import Agent

from .board import CellState, WallState, Direction


class Firefighter(Agent):
    STARTING_AP = 4

    def __init__(self, model):
        super().__init__(model)
        self.ap = self.STARTING_AP
        self.poss: tuple[int, int] | None = None

    @property
    def board(self):
        return self.model.board

    # ---- actions ------------------------------------------------------

    def move(self, direction: Direction) -> bool:
        # here we move
        return True

    def move_with_victim(self, direction: Direction) -> bool:
        # here we move with the victim
        return True

    def open_close_door(self, direction: Direction) -> bool:
        # here we open and close doors
        return True

    def extinguish(self, direction: Direction | None = None) -> bool:
        """direction=None targets the firefighter's own cell."""
        # here we extinguish fire
        return True

    def chop(self, direction: Direction) -> bool:
        # here we chop walls
        return True

    def step(self):
        # here we code the decision making of the agent
        # i think that in every turn of the agent it shoud
        # make its action
        # advance fire
        # and replenish pois
        pass
