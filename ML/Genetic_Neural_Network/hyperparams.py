N_INNER = 2  # 은닉층 뉴런 개수
BRAIN_SIZE = 5  # 최대 connection 개수
P_MUTATION = 0.04  # 변이 확률
MAX_WEIGHT = 10.0

k_velocity = 0.2

CANVAS_DIM = 720
CANVAS_PAD = 30
SETWIN_WIDTH = 500
PLOTWIN_HEIGHT = 500
FRAMES_PER_GEN = 300
MS_PER_FRAME = 10
INITIAL_GEN_POP = 900
INITIAL_POS = []  # the initial positions at the start of each gen
for i in range(30):
        for j in range(30):
            INITIAL_POS.append((5.0+i*20.0, 5.0+j*20.0))