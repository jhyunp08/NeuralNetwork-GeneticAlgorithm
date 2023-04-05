N_PER_LAYER = 2  # 은닉층 뉴런 개수
BRAIN_SIZE = 7  # 최대 connection 개수
BRAIN_DEPTH = 2 # hidden layer 개수 < 16 - 1
P_MUTATION = 0.035  # 변이 확률
MAX_WEIGHT = 10.0

K_VELOCITY = 0.3

CANVAS_DIM = 720
CANVAS_PAD = 30
SETWIN_WIDTH = 500
PLOTWIN_HEIGHT = 500
FRAMES_PER_GEN = 200
INITIAL_GEN_POP = 900

INITIAL_POS = []  # the initial positions at the start of each gen
for i in range(30):
        for j in range(30):
            INITIAL_POS.append((75.0+i*19.0, 22.5+j*22.5))
GOAL_ZONE = [
      ((0, 72), (10, 710)), 
      ((648, 720), (10, 710))
]
DEAD_ZONE = [
      ((0, 72), (0, 10)),
      ((0, 72), (710, 720)),
      ((648, 720), (0, 10)),
      ((648, 720), (710, 720))
]
WALLS = [
      ((70, 75), (100, 200))
]

ms_per_frame = 25
frameCount = 0


class Entities(list):
    def __init__(self):
            list.__init__(self)
            self.alive = 0
    def append(self, __object, alive=True) -> None:
          if alive:
               self.alive += 1
          return super().append(__object)

entities = Entities()  # all entities will be stored and accessed through this list
entities_dead = []