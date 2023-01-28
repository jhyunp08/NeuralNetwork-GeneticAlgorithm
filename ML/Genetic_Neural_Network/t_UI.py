# test file; tkinter GUI
import random as rand
import numpy as np
from functools import cache, lru_cache
import asyncio  # managing coroutines
import networkx as nx  # generating graphs
import matplotlib.pyplot as plt  # visualizing graphs
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import PIL  # image processing

k_velocity = 0.1

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def null(x):
    return 0.0


def llun(x):
    return 1.0

k_vel = 5
def move(ent, mvmnt):
    ddd = {'n': (0, -1), 's': (0, 1), 'e': (1, 0), 'w': (-1, 0)}
    ent.x += ddd[mvmnt][0]
    ent.y += ddd[mvmnt][1]

entities = []

class Entity:
    def __init__(self, x, y) -> None:
        self.id = len(entities)

        self.x = x
        self.y = y
        self.obj = None
    
    def __eq__(self, __o: object) -> bool:
        try:
            return (self.id == __o.id)
        except AttributeError:
            print(f'Attribute Error: {__o} is not of class Entity')
        finally:
            return False
    
    def dist(self, e) -> float:
        return np.sqrt((self.x - e.x) ** 2 + (self.y - e.y) ** 2)

    def nearest(self, _return_dist=False):
        min_i = 0
        min_dist = 10000.0
        for i_e in range(len(entities)):
            if i_e == self.id:
                continue
            d_i = self.dist(entities[i_e])
            if self.dist(entities[i_e]) < min_dist:
                min_i = i_e
                min_dist = d_i
        if _return_dist:
            return min_dist
        return entities[min_i]

    def mutate(self):
        pass

def setUp():
    for i in range(30):
        for j in range(30):
            entities.append(Entity(5.0+(i)*20.0, 5.0+(j)*20.0))


import platform
from tkinter import *
from tkinter import font as tkFont

if platform.system() == "Darwin":
    from tkmacosx import Button
else:
    from tkinter import Button as tkButton


    class Button(tkButton):
        def __init__(self, master=None, cnf={}, **kw):
            kwargs = kw
            if "height" in kw:
                kwargs["height"] = int(kw["height"] / 30)
            if "width" in kw:
                kwargs["width"] = int(kw["width"] / 14)
            tkButton.__init__(self, master, cnf, **kwargs)


def set_default_font():
    def_font = tkFont.nametofont("TkDefaultFont")
    def_font.config(family="Helvetica", size=15)


CANVAS_DIM = 720
CANVAS_PAD = 30
SETWIN_WIDTH = 500


class App(Tk):
    def __init__(self, master=None):
        # initialize window
        Tk.__init__(self)
        self.wm_title("hello world?")
        self.geometry(f"{CANVAS_DIM + 2*CANVAS_PAD + SETWIN_WIDTH}x{CANVAS_DIM + 2*CANVAS_PAD}")
        self.menu = Menu(self)
        self.config(menu=self.menu)
        self.createMenu()

        self.style = [{"bg": "gray20", "fg": "#bbdbef", "canvas_bg": "gray6", "canvas_fg": "white"},
                      {"bg": "gray15", "highlightbackground": "#404040", "highlightthickness": 2, "fg": "white", 
                      "pbut_bg": "gray", "rbut_bg": "gray17"}]
        self.fonts = {"gen": tkFont.Font(family='times', size=18)}

        self.sim_win = SimWindow(self)
        self.set_win = SettingWindow(self)
    
    def createMenu(self):
        fileMenu = Menu(self.menu)
        fileMenu.add_command(label="Exit", command=self.exitProgram)
        self.menu.add_cascade(label="File", menu=fileMenu)

        helpMenu = Menu(self.menu)
        self.menu.add_cascade(label="Help", menu=helpMenu)

    def exitProgram(self):
        exit()


class SimWindow(Frame):
    def __init__(self, master):
        Frame.__init__(self, master, width=CANVAS_DIM + 2*CANVAS_PAD, height=CANVAS_DIM + 2*CANVAS_PAD)
        self.master = master
        self.pack(side=LEFT, fill=BOTH, expand=1)
        self.style_dict = self.master.style[0]
        self.config(bg=self.style_dict["bg"])

        self.canvas = Canvas(self, width=self.cget("width") - 2*CANVAS_PAD, height=self.cget("height") - 2*CANVAS_PAD,
                             bg=self.style_dict["canvas_bg"], highlightthickness=0)
        self.canvas.place(x=CANVAS_PAD, y=CANVAS_PAD)

        self.genCount = Label(self, text="Gen: 0", font=self.master.fonts["gen"], fg=self.style_dict["fg"], bg=self.style_dict["bg"])
        self.genCount.pack(side=TOP)

        self.draw()
        self.bind('<Button-1>', self.f)
    
    def draw(self):
        for e in entities:
            e.obj = self.canvas.create_oval(e.x-1, e.y-1, e.x+1, e.y+1, fill=self.style_dict["fg"], tags="entity")
        self.update()

    def update_(self):
        for e in entities:
            self.canvas.moveto(e.obj, e.x, e.y)
        self.update()

    def f(self, event):
        print(1)
        for e in entities:
            move(e, 's')
        self.update_()


class SettingWindow(Frame):
    def __init__(self, master):
        Frame.__init__(self, master, width=SETWIN_WIDTH)
        self.master = master
        self.pack(side=RIGHT, fill=Y, expand=0)
        self.style_dict = self.master.style[1]
        self.config(bg=self.style_dict["bg"], highlightbackground=self.style_dict["highlightbackground"],
                    highlightthickness=self.style_dict["highlightthickness"])

        self.playpauseButton = Button(self,
                                  text="Play",
                                  width=90,
                                  height=25,
                                  fg=self.style_dict["fg"],
                                  bg=self.style_dict["pbut_bg"],
                                  highlightbackground=self.style_dict["bg"],
                                  command=None
                                  )
        self.playpauseButton.place(x=10, y=50)
        
        self.resetButton = Button(self,
                                  text="Reset",
                                  width=90,
                                  height=25,
                                  fg=self.style_dict["fg"],
                                  bg=self.style_dict["rbut_bg"],
                                  highlightbackground=self.style_dict["bg"],
                                  command=None
                                  )
        self.resetButton.place(x=120, y=50)


        self.figure_pop = Figure(figsize=(4, 2), dpi=100)

        self.plot_pop_1 = self.figure_pop.add_subplot(1, 1, 1)
        self.plot_pop_1.set_title("Population")
        self.plot_pop_1.set_xlabel("gen")
        self.plot_pop_1.set_xlim(0, 100)
        self.plot_pop_1.plot([np.sin(i) + 100 for i in range(100)], 'b')

        self.graph_pop = FigureCanvasTkAgg(self.figure_pop, master = self)
        self.graph_pop.draw()
        self.graph_pop.get_tk_widget().place(x=10, y=200)


        self.figure_gene = Figure(figsize=(4, 2), dpi=100)

        self.plot_gene_1 = self.figure_gene.add_subplot(1, 1, 1)
        self.plot_gene_1.set_title("Gen Pool")
        self.plot_gene_1.set_xlabel("gen")
        self.plot_gene_1.set_xlim(0, 100)
        self.plot_gene_1.plot([100-i for i in range(100)], 'r')

        self.graph_gene = FigureCanvasTkAgg(self.figure_gene, master = self)
        self.graph_gene.draw()
        self.graph_gene.get_tk_widget().place(x=10, y=450)


if __name__ == "__main__":
    setUp()
    app = App()
    set_default_font()
    app.mainloop()
