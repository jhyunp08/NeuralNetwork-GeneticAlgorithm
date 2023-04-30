import numpy as np
PI = np.pi

N_PER_LAYER = 3  # 은닉층 뉴런 개수
BRAIN_SIZE = 12  # 최대 connection 개수
BRAIN_DEPTH = 2 # hidden layer 개수 < 16 - 1
P_MUTATION = 0.04  # 변이 확률
MAX_WEIGHT = 10.0

K_VELOCITY = 1.2

CANVAS_DIM = 720
CANVAS_PAD = 30
SETWIN_WIDTH = 500
PLOTWIN_HEIGHT = 600
FRAMES_PER_GEN = 200
INITIAL_GEN_POP = 900

INITIAL_POS = []  # the initial positions at the start of each gen
for i in range(INITIAL_GEN_POP):
      INITIAL_POS.append((min(520, max(200, np.random.normal(CANVAS_DIM/2, 70))), min(539, max(181, np.random.normal(CANVAS_DIM/2, 70)))))
GOAL_ZONE = [
      ((0, 150), (0, 150)),
      ((0, 150), (570, 720)),
      ((570, 720), (0, 150)),
      ((570, 720), (570, 720))
]
DEAD_ZONE = [
      ((0, 72), (220, 500)), 
      ((648, 720), (220, 500))
]
WALLS = [
      ((200, 520), (170, 180)),
      ((200, 520), (540, 550)),
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