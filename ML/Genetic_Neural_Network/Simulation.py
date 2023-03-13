# code for the simulation
from classes import *
from hyperparams import *
import platform
from tkinter import *
from tkinter import font as tkFont

# using tkmacosx as the current version of tkinter doesn't support displaying button widget with background color in macosx
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


class Root(Tk):
    def __init__(self):
        # initialize window
        Tk.__init__(self)
        self.wm_title("hello world?")
        self.geometry(f"{CANVAS_DIM + 2*CANVAS_PAD + SETWIN_WIDTH}x{CANVAS_DIM + 2*CANVAS_PAD}")
        self.menu = Menu(self)
        self.config(menu=self.menu)
        self.createMenu()

        self.style = [
            {"bg": "gray20", "fg": "#bbdbef", "canvas_bg": "gray6", "canvas_fg": "white"},
            {"bg": "gray15", "highlightbackground": "#404040", "highlightthickness": 2, 
                "fg": "white", "pbut_bg": "gray", "rbut_bg": "gray17"},
            {"bg": "gray15", "highlightbackground": "#404040", "highlightthickness": 2, "fg": "white"}
            ]
        self.fonts = {"gen": tkFont.Font(family='times', size=18)}

        self.sim_win = SimWindow(self)
        self.set_win = SettingWindow(self)
        self.plot_win = PlotWindow(self)
    
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

        self.gen = 0
        self.genCount = Label(self, font=self.master.fonts["gen"], fg=self.style_dict["fg"], bg=self.style_dict["bg"], text="")
        self.genCount.pack(side=TOP)
        self.running = False

        self.setUp()
    
    def setUp(self):
        # initilaize all entities and assign canvas object oval
        for i in range(INITIAL_GEN_POP):
            e = Entity(0, 0)
            e.network = Network(e, N_INNER, randGene())
            entities.append(e)
        for e in entities:
            e.obj = self.canvas.create_oval(e.x-1, e.y-1, e.x+1, e.y+1, fill=self.style_dict["fg"], tags="entity")

    def update_(self):
        for e in entities:
            self.canvas.moveto(e.obj, e.x-1, e.y-1)
        self.update()

    def place_entities(self):
        rand.shuffle(entities)
        for i, e in enumerate(entities):
            e.x, e.y = INITIAL_POS[i]
        self.update_()

    def run(self):
        for e in entities:
            e.run() ###
        self.update_()

    def loop_frames(self):
        if not self.running:
            return
        global frameCount
        if frameCount >= FRAMES_PER_GEN:
            self.gen += 1
            frameCount = 0
            self.newGen()
            return
        self.run()
        frameCount += 1
        self.after(ms_per_frame, self.loop_frames)

    def newGen(self):
        self.genCount.config(text=f"Gen: {self.gen}")
        if self.gen > 0:
            alive_genes = []
            new_genes = []
            for e in entities:
                if e.alive:
                    alive_genes.append(e.network.gene)
                e.reset()
            if alive_genes:
                rand.shuffle(alive_genes)
                i = 0
                while i < INITIAL_GEN_POP:
                    new_genes.append(mutateGene(alive_genes[i % len(alive_genes)]))
                    i += 1
                for i in range(INITIAL_GEN_POP):
                    e = entities[i]
                    e.network = Network(e, N_INNER, new_genes[i])
            else:
                print("Simulation Terminated; No Entities Alive")
                return
        self.place_entities()
        self.loop_frames()
    
    def start_simulation(self):
        self.running = True
        self.newGen()
        self.master.set_win.playpauseButton.config(text="Pause", command=lambda: self.master.set_win.after(1, self.pause_simulation))

    def restart_simulation(self):
        self.running = True
        self.loop_frames()
        self.master.set_win.playpauseButton.config(text="Pause", command=lambda: self.master.set_win.after(1, self.pause_simulation))

    def pause_simulation(self):
        self.running = False
        self.master.set_win.playpauseButton.config(text="Play", command=lambda: self.master.set_win.after(1, self.restart_simulation))


class SettingWindow(Frame):
    def __init__(self, master):
        Frame.__init__(self, master, width=SETWIN_WIDTH, height=CANVAS_DIM+2*CANVAS_PAD-PLOTWIN_HEIGHT)
        self.master = master
        self.pack(side=TOP, expand=1)
        self.style_dict = self.master.style[1]
        self.config(bg=self.style_dict["bg"], highlightbackground=self.style_dict["highlightbackground"],
                    highlightthickness=self.style_dict["highlightthickness"])

        self.playpauseButton = Button(
            self,
            text="Play",
            width=90,
            height=25,
            fg=self.style_dict["fg"],
            bg=self.style_dict["pbut_bg"],
            highlightbackground=self.style_dict["bg"],
            command=lambda: self.after(1, self.master.sim_win.start_simulation)
        )
        self.playpauseButton.place(x=10, y=50)
        
        self.resetButton = Button(
            self,
            text="Reset",
            width=90,
            height=25,
            fg=self.style_dict["fg"],
            bg=self.style_dict["rbut_bg"],
            highlightbackground=self.style_dict["bg"],
            command=None
        )
        self.resetButton.place(x=120, y=50)

        self.label_timeScale = Label(
            self,
            text="time \u00d7:",
            fg=self.style_dict["fg"],
            bg=self.style_dict["bg"]
        )
        self.label_timeScale.place(x=25, y=100)

        self.timeScale = Scale(
            self,
            from_=0.1,
            to=15,
            tickinterval=2,
            length=400,
            orient=HORIZONTAL
        )
        self.timeScale.set(1.0)
        self.timeScale.place(x=35, y=130)


class PlotWindow(Frame):
    def __init__(self, master):
        Frame.__init__(self, master, width=SETWIN_WIDTH, height=PLOTWIN_HEIGHT)
        self.master = master
        self.pack(side=RIGHT, expand=0)
        self.style_dict = self.master.style[2]
        self.config(bg=self.style_dict["bg"], highlightbackground=self.style_dict["highlightbackground"],
                    highlightthickness=self.style_dict["highlightthickness"])

        self.figure_pop = Figure(figsize=(4, 2), dpi=100)

        self.plot_pop_1 = self.figure_pop.add_subplot(1, 1, 1)
        self.plot_pop_1.set_title("Population")
        self.plot_pop_1.set_xlabel("gen")
        self.plot_pop_1.set_xlim(0, 100)
        self.plot_pop_1.plot([np.sin(i) + 100 for i in range(100)], 'b')

        self.graph_pop = FigureCanvasTkAgg(self.figure_pop, master = self)
        self.graph_pop.draw()
        self.graph_pop.get_tk_widget().place(x=10, y=10)

        self.figure_gene = Figure(figsize=(4, 2), dpi=100)

        self.plot_gene_1 = self.figure_gene.add_subplot(1, 1, 1)
        self.plot_gene_1.set_title("Gen Pool")
        self.plot_gene_1.set_xlabel("gen")
        self.plot_gene_1.set_xlim(0, 100)
        self.plot_gene_1.plot([100-i for i in range(100)], 'r')

        self.graph_gene = FigureCanvasTkAgg(self.figure_gene, master = self)
        self.graph_gene.draw()
        self.graph_gene.get_tk_widget().place(x=10, y=250)



if __name__ == "__main__":
    root = Root()
    set_default_font()
    root.mainloop()
