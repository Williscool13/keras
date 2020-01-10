import tkinter as tk
from .engine.input_layer import Input

GLOBAL_STRUCTURE = []

class display:
    def __init__(self, master):
        self.master = master
        self.frame = tk.Frame(self.master)
        self.button1 = tk.Button(self.frame, text = 'Create Model', width=25, command = self.model_creation)
        self.button1.pack()
        self.frame.pack()


    def model_creation(self):
        self.new_window = tk.Toplevel(self.master)
        self.show = model_build(self.new_window)


class model_build:
    def __init__(self, master):
        self.master = master
        self.frame = tk.Frame(self.master)
        self.label.pack()
        self.frame.pack()
        self.button = tk.Button(self.frame, text='Input', command=self.input_size)
        self.button.pack()


    def input_size(self):
        self.new_window = tk.Toplevel(self.master)
        self.temp = input_window(self.new_window)

class input_window:
    def __init__(self, master):
        self.master = master
        self.temp = tk.Frame(self.master)
        self.label1 = tk.Label(self.temp, text='Dimensions:')
        self.label2 = tk.Label(self.temp, text='Input Channels:')
        self.input1 = tk.Entry(self.temp)
        self.input2 = tk.Entry(self.temp)
        button = tk.Button(self.temp, text='Ok', command = self.get_input)
        self.label1.pack()
        self.input1.pack()
        self.label2.pack()
        self.input2.pack()
        button.pack()
        self.temp.pack()

    def get_input(self):
        input_size = (self.input1.get(), self.input1.get(), self.input2.get())
        GLOBAL_STRUCTURE.append(Input(shape=input_size))
        tk.Label(self.temp, text='input successfully saved').pack()



def main():
    root = tk.Tk()
    face = display(root)
    root.mainloop()
    print(GLOBAL_STRUCTURE)

if __name__ == '__main__':
    main()
