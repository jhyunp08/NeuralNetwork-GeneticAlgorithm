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

        self.style = {
            "sim": {"bg": "gray20", "fg": "#bbdbef", "canvas_bg": "gray6", "canvas_fg": "white",
                    "goal": "#07c142", "dead": "#b33535", "wall": "gray"},
            "set": {"bg": "gray15", "highlightbackground": "#404040", "highlightthickness": 2,
                    "fg": "white", "pbut_bg": "gray", "rbut_bg": "gray17"},
            "plot": {"bg": "gray15", "highlightbackground": "#404040", "highlightthickness": 2, "fg": "white"}
        }
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
        self.style_dict = self.master.style["sim"]
        self.config(bg=self.style_dict["bg"])

        self.canvas = Canvas(self, width=self.cget("width") - 2*CANVAS_PAD, height=self.cget("height") - 2*CANVAS_PAD,
                             bg=self.style_dict["canvas_bg"], highlightthickness=0)
        self.canvas.place(x=CANVAS_PAD, y=CANVAS_PAD)

        self.gen = 0
        self.genCount = Label(self, font=self.master.fonts["gen"], fg=self.style_dict["fg"], bg=self.style_dict["bg"], text="")
        self.genCount.pack(side=TOP)

        self.survived = []
        self.running = False

        self.setUpEnv()
    
    def setUpEnv(self):
        for rectangle in GOAL_ZONE:
            self.canvas.create_rectangle(rectangle[0][0], rectangle[1][0], rectangle[0][1], rectangle[1][1],
                                         fill=self.style_dict["goal"], outline=self.style_dict["goal"], tags="goal")
        for rectangle in DEAD_ZONE:
            self.canvas.create_rectangle(rectangle[0][0], rectangle[1][0], rectangle[0][1], rectangle[1][1],
                                         fill=self.style_dict["dead"], outline=self.style_dict["dead"], tags="dead")
        for rectangle in WALLS:
            self.canvas.create_rectangle(rectangle[0][0], rectangle[1][0], rectangle[0][1], rectangle[1][1],
                                         fill=self.style_dict["wall"], outline=self.style_dict["wall"], tags="wall")
    
    def setUpEnt(self):
        for i in range(INITIAL_GEN_POP):
            e = Entity(0, 0)
            e.network = Network(e, N_INNER, randGene())
            entities.append(e)
        for e in entities:
            e.obj = self.canvas.create_oval(e.x-1, e.y-1, e.x+1, e.y+1, fill=self.style_dict["fg"], tags="entity")

    def resetUp(self):
        self.gen = 0
        self.survived.clear()
        global frameCount
        frameCount = 0
        entities.clear()
        entities_dead.clear()
        self.canvas.delete("entity")
        self.master.plot_win.reset()

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
            surviving_genes = []
            new_genes = []
            for e in entities:
                if e.inRectangles(GOAL_ZONE):
                    surviving_genes.append(e.network.gene)
                e.reset()
            for e in entities_dead:
                entities.append(e)
            entities_dead.clear()
            self.survived.append(surviving_genes)
            self.master.plot_win.plot_point("surv_rate", self.gen-1, len(surviving_genes), 'b')
            if surviving_genes:
                rand.shuffle(surviving_genes)
                i = 0
                while i < INITIAL_GEN_POP:
                    new_genes.append(mutateGene(surviving_genes[i % len(surviving_genes)]))
                    i += 1
                for i in range(INITIAL_GEN_POP):
                    e = entities[i]
                    e.network = Network(e, N_INNER, new_genes[i])
            else:
                self.pause_simulation()
                print("Simulation Terminated; No Entities Alive")
                return
        self.place_entities()
        self.loop_frames()
    
    def start_simulation(self):
        self.setUpEnt()
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

    def reset_simulation(self):
        self.running = False
        self.after(1, self.resetUp())
        self.master.set_win.playpauseButton.config(text="Play", command=lambda: self.master.set_win.after(1, self.start_simulation))


class SettingWindow(Frame):
    def __init__(self, master):
        Frame.__init__(self, master, width=SETWIN_WIDTH, height=CANVAS_DIM+2*CANVAS_PAD-PLOTWIN_HEIGHT)
        self.master = master
        self.pack(side=TOP, expand=1)
        self.style_dict = self.master.style["set"]
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
            command=lambda: self.after(1, self.master.sim_win.reset_simulation)
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
        self.style_dict = self.master.style["plot"]
        self.config(bg=self.style_dict["bg"], highlightbackground=self.style_dict["highlightbackground"],
                    highlightthickness=self.style_dict["highlightthickness"])
        parameters = {
            'axes.labelsize': 7,
            'xtick.labelsize': 5,
            'ytick.labelsize': 7,
            'axes.titlesize': 10
        }
        plt.rcParams.update(parameters)

        self.figure_pop = Figure(figsize=(4, 2), dpi=100)
        self.graph_pop = FigureCanvasTkAgg(self.figure_pop, master = self)
        
        self.figure_gene = Figure(figsize=(4, 2), dpi=100)
        self.graph_gene = FigureCanvasTkAgg(self.figure_gene, master = self)

        self.set_plots()

        self.graph_pop.draw()
        self.graph_pop.get_tk_widget().place(x=10, y=10)

        self.graph_gene.draw()
        self.graph_gene.get_tk_widget().place(x=10, y=250)

    def set_plots(self):
        self.plot_surv_rate = self.figure_pop.add_subplot(1, 1, (1,1))
        self.plot_surv_rate.set_title("Survival Rate")
        self.plot_surv_rate.set_xlabel("gen", labelpad=0)
        self.plot_surv_rate.set_xbound(0, 2)
        self.plot_surv_rate.set_ybound(0, INITIAL_GEN_POP)

        self.plot_gene_1 = self.figure_gene.add_subplot(1, 1, 1)
        self.plot_gene_1.set_title("Gene Pool")
        self.plot_gene_1.set_xlabel("gen")
    
    def reset(self):
        self.figure_pop.clear()
        self.figure_gene.clear()
        self.set_plots()
        self.graph_pop.draw()
        self.graph_gene.draw()

    def plot_point(self, plot, x, y, color):
        plot_dict = {
            "surv_rate": (self.graph_pop, self.plot_surv_rate),
            "gene_1": (self.graph_gene, self.plot_gene_1)
        }
        plot_dict[plot][1].scatter([x], [y], color=color)
        plot_dict[plot][0].draw()


if __name__ == "__main__":
    root = Root()
    set_default_font()
    root.mainloop()
