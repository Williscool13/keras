import tkinter as tk
import time
from .engine.input_layer import Input
from .layers.convolutional import Conv2D

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
        self.button = tk.Button(self.frame, text='Input', command=self.input)
        self.button1= tk.Button(self.frame, text='Conv2D', command=self.conv2d)

        self.button.pack()
        self.button1.pack()
        self.frame.pack()


    def input(self):
        new_window = tk.Toplevel(self.master)
        temp = input_window(new_window)

    def conv2d(self):
        #need to add activation to the list of hyperparams
        new_window = tk.Toplevel(self.master)
        temp = conv2d_window(new_window)


class conv2d_window:
    def __init__(self, master):
        self.master = master
        self.temp = tk.Frame(self.master)
        self.filters        = tk.Label(self.temp, text='Number of filters:')
        self.filters_input  = tk.Entry(self.temp)
        self.kernel         = tk.Label(self.temp, text='Kernel size:')
        self.kernel_input   = tk.Entry(self.temp)
        self.stride         = tk.Label(self.temp, text='Stride size:')
        self.stride_input   = tk.Entry(self.temp)
        self.padding        = tk.Label(self.temp, text='Stride type:')

#       self.padding_menu  = tk.Menubutton(self.temp, text='stride')
#        self.padding_menu.menu = tk.Menu(self.padding_input)
#        self.padding_menu['menu'] = self.padding_menu.menu
#        self.padding_menu.menu.add_checkbutton(label = 'valid', command=self.set_valid)
#        self.padding_menu.menu.add_checkbutton(label = 'same',  command=self.set_same)

        padding_options = ['same', 'valid']
        self.padding_value = tk.StringVar(self.temp)
        self.padding_value.set(padding_options[0])

        self.padding_menu = tk.OptionMenu(self.temp, self.padding_value, *padding_options)


        self.filters.pack()
        self.filters_input.pack()
        self.kernel.pack()
        self.kernel_input.pack()
        self.stride.pack()
        self.stride_input.pack()
        self.padding.pack()
        self.padding_menu.pack()

        self.warning_text   = tk.StringVar()
        self.warning_label  = tk.Label(self.temp, textvariable=self.warning_text)
        self.warning_label.pack()

        self.button = tk.Button(self.temp, text='Ok', command = self.get_conv2d)
        self.button.pack()

        self.temp.pack()

    def get_conv2d(self):
        if self.filters_input.get().isdigit():
            filter = int(self.filters_input.get())
        else:
            self.warning_text.set('Filter Size Invalid')
            return

        if self.kernel_input.get().isdigit():
            kernel = int(self.kernel_input.get())
        else:
            self.warning_text.set('Kernel Size Invalid')
            return

        if self.stride_input.get().isdigit():
            stride = int(self.stride_input.get())
        else:
            self.warning_text.set('Stride Size Invalid')
            return

        padding = self.padding_value.get()
        op = Conv2D(filters=filter, kernel_size=kernel, strides=stride, padding=padding)
        GLOBAL_STRUCTURE.append(op)
        tk.Label(self.temp, text='Input Successfully Saved').pack()
        self.temp.after(1500, self.master.destroy)

class input_window:
    def __init__(self, master):
        self.master = master
        self.temp = tk.Frame(self.master)
        self.label1 = tk.Label(self.temp, text='Dimensions:')
        #self.label2 = tk.Label(self.temp, text='Input Channels:')
        self.input1 = tk.Entry(self.temp)
        #self.input2 = tk.Entry(self.temp)
        self.button = tk.Button(self.temp, text='Ok', command = self.get_input)
        self.label1.pack()
        self.input1.pack()
        #self.label2.pack()
        #self.input2.pack()
        self.button.pack()
        self.temp.pack()

    def get_input(self):
        i = tuple([int(x.strip()) for x in self.input1.get().split(',')])
        GLOBAL_STRUCTURE.append(Input(shape=i))
        self.button.destroy()
        self.label1.destroy()
        self.input1.destroy()
        tk.Label(self.temp, text='Dimensions: {}'.format(i)).pack()
        tk.Label(self.temp, text='input successfully saved').pack()
        self.temp.after(1500, self.master.destroy)
        #self.master.destroy()


from .models import Sequential
from .models import Model
def main():
    root = tk.Tk()
    face = display(root)
    root.mainloop()

    if len(GLOBAL_STRUCTURE) > 1:

        x = GLOBAL_STRUCTURE[0]
        for item in GLOBAL_STRUCTURE[1:-1]:
            x = item(x)
        y = GLOBAL_STRUCTURE[-1](x)
        model = Model(x,y)
        print(model.summary())

    else:
        print('only 1 layer in structure, unable to create model')

if __name__ == '__main__':
    main()
