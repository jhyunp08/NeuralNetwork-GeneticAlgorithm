import numpy as np

N_PER_LAYER = 3  # 은닉층 뉴런 개수
BRAIN_SIZE = 12  # 최대 connection 개수
BRAIN_DEPTH = 2 # hidden layer 개수 < 16 - 1
P_MUTATION = 0.035  # 변이 확률
MAX_WEIGHT = 10.0

K_VELOCITY = 1.2

CANVAS_DIM = 720
CANVAS_PAD = 30
SETWIN_WIDTH = 500
PLOTWIN_HEIGHT = 600
FRAMES_PER_GEN = 250
INITIAL_GEN_POP = 900

INITIAL_POS = []  # the initial positions at the start of each gen
for i in range(INITIAL_GEN_POP):
      INITIAL_POS.append((min(CANVAS_DIM, max(0, np.random.normal(CANVAS_DIM/2, 70))), min(CANVAS_DIM, max(0, np.random.normal(CANVAS_DIM/2, 70)))))
GOAL_ZONE = [
      ((0, 72), (200, 520)), 
      ((648, 720), (200, 520))
]
DEAD_ZONE = [
      ((0, 72), (0, 100)),
      ((0, 72), (620, 720)),
      ((648, 720), (0, 100)),
      ((648, 720), (620, 720))
]
WALLS = [
      ((180, 540), (180, 190)),
      ((180, 540), (530, 540)),
]

frame_delay_ms = 25
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