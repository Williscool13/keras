import tkinter as tk
from tkinter import filedialog
import time
import os
import pandas as pd

from .models import Sequential, Model
from .engine.input_layer import Input
from .layers.convolutional import Conv1D, Conv2D, Conv3D, SeparableConv1D, SeparableConv2D, DepthwiseConv2D, Conv2DTranspose, Conv3DTranspose
from .blocks import Inception, Residual, VGG, SqueezeExcite, InvertedResidual
from .models import Sequential
from .losses import *
from .optimizers import *
from .activations import *
from .layers.core import Dense

INPUT_SHAPE = []
GLOBAL_STRUCTURE = []
GLOBAL_LOSS = [0]
GLOBAL_OPTIMIZER = [0]
GLOBAL_PARAMS = [0]
GLOBAL_MODEL = [0]


class display:
    def __init__(self, master):
        self.master = master
        self.master.title('Main Window')
        self.frame = tk.Frame(self.master)

        tk.Button(self.frame, text='Build Model',               width=40, height=5, command = self.build_model).grid(row=0, column=0)
        tk.Button(self.frame, text='Train Model',               width=40, height=5, command = self.train_model).grid(row=1, column=0)
        tk.Button(self.frame, text='Exit',                      width=40, height=5, command = lambda:quit(self.master)).grid(row=2, column=0)

        self.frame.pack()


    def build_model(self):
        new_window = tk.Toplevel(self.master)
        Build(new_window)

    def train_model(self):
        new_window = tk.Toplevel(self.master)
        Train(new_window)


class Train:
    def __init__(self, master):
        self.master = master
        self.master.title('Model Training')
        self.frame = tk.Frame(self.master)
        self.INPUT = tk.Button(self.frame, text = 'Input Data',              width=40, height=5, command = self.get_input_file, bg='red')
        self.INPUT.grid(row=0, column=0)
        self.OPT   = tk.Button(self.frame, text = 'Optimizer' ,              width=40, height=5, command = self.get_optimizer,  bg='red')
        self.OPT.grid(row=1, column=0)
        self.LOSS  = tk.Button(self.frame, text = 'Loss'      ,              width=40, height=5, command = self.get_loss,       bg='red')
        self.LOSS.grid(row=2, column=0)
        tk.Label(self.frame, text='Epochs:')
        self.TRAIN_PARAMS = tk.Button(self.frame, text='Training Parameters',width=40, height=5, command = self.get_params,     bg='red')
        self.TRAIN_PARAMS.grid(row=3, column=0)
        self.TRAIN = tk.Button(self.frame, text='TRAIN!',                    width=40, height=5, command = self.train)
        self.TRAIN.grid(row=4, column=0)
        tk.Button(self.frame, text='Predict!',                               width=40, height=5, command = self.predict).grid(row=5, column=0)


        tk.Button(self.frame, text = 'Exit',                    width=40, height=5, command = lambda:quit(self.master)).grid(row=6, column=0)
        self.frame.pack()

    def get_input_file(self):
        currdir = os.getcwd()
        tempdir = filedialog.askopenfilename(parent=self.master, initialdir=currdir, title='Please select a directory')
        if tempdir.endswith('.csv'):
            self.INPUT.configure(bg='green')
            df = pd.read_csv(tempdir)
            self.y = df['y']
            self.X = df.drop('y', axis=1)
        else:
            dummy = tk.Toplevel(self.master)
            dummy.title('Error')
            tk.Label(dummy, text='Incorrect filetype', font=(36), width=30).pack()
            self.master.after(1500, dummy.destroy)



    def get_optimizer(self):
        new_window = tk.Toplevel(self.master)
        Optimizer(new_window, self)

    def get_loss(self):
        new_window = tk.Toplevel(self.master)
        Loss(new_window, self)

    def get_params(self):
        new_window = tk.Toplevel(self.master)
        Params(new_window, self)

    def train(self):
        assemble()
        GLOBAL_MODEL[0].compile(optimizer=GLOBAL_OPTIMIZER[0], loss=GLOBAL_LOSS[0])
        GLOBAL_MODEL[0].fit(self.X, self.y, batch_size=int(GLOBAL_PARAMS[0][0]), epochs=int(GLOBAL_PARAMS[0][1]))

    def predict(self):
        pred = GLOBAL_MODEL[0].predict([[self.X.iloc[0].to_numpy()]])
        window = tk.Toplevel(self.master)
        s = f'Input: {self.X.iloc[0].to_numpy()}\nPredicted: {pred}\nReal: {self.y.iloc[0]}'
        tk.Label(window, text=s).pack()
        window.after(5000, window.destroy)


class Params:
    def __init__(self, master, controller):
        self.master = master
        self.master.title('Training Parameters')
        self.controller = controller
        tk.Label(self.master, text='Batch Size:').pack()
        self.bs = tk.Entry(self.master)
        self.bs.pack()

        tk.Label(self.master, text='Epochs:').pack()
        self.eps = tk.Entry(self.master)
        self.eps.pack()

        self.button = tk.Button(self.master, text='Ok', command = self.get_params, width=20)
        self.button.pack()

    def get_params(self):
        GLOBAL_PARAMS[0] = (self.bs.get(), self.eps.get())
        self.controller.TRAIN_PARAMS.configure(bg='green')
        self.master.destroy()




class Loss:
    def __init__(self, master, controller):
        self.master = master
        self.master.title('Loss')
        self.controller = controller
        losses = ['Mean Squared Error', 'Mean Absolute Error', 'Mean Absolute Percentage Error', 'Mean Squared Logarithmic Errors', 'Squared Hinge', 'Hinge', 'Categorical Hinge', 'Logcosh']
        self.LOSS = {
                "Mean Squared Error": 'mean_squared_error',
                "Mean Absolute Error": 'mean_absolute_error',
                "Mean Absolute Percentage Error": 'mean_absolute_percentage_error',
                "Mean Squared Logarithmic Error": 'mean_squared_logarithmic_error',
                "Squared Hinge": 'squared_hinge',
                "Hinge": 'hinge',
                "Categorical Hinge": 'categorical_hinge',
                "LogCosh": 'logcosh'
                }
        self.v = tk.StringVar(None, 'Mean Squared Error')
        for name in losses:
            tk.Radiobutton(self.master, text=name, variable=self.v, value=name).pack()

        self.button = tk.Button(self.master, text='Ok', command = self.get_loss, width=20)
        self.button.pack()

    def get_loss(self):
        opt = self.LOSS[self.v.get()]
        GLOBAL_LOSS[0] = opt
        self.controller.LOSS.configure(text=self.v.get(), bg='green')
        self.master.destroy()


class Optimizer:
    def __init__(self, master, controller):
        self.master = master
        self.master.title('Optimizer')
        self.controller = controller

        optimizers = ['SGD', 'RMSprop', 'Adagrad', 'Adam', 'Adamax', 'Nadam']
        self.OPS = {'SGD': SGD, 'RMSprop': RMSprop, 'Adagrad': Adagrad, 'Adam': Adam, 'Adamax': Adamax, 'Nadam': Nadam}

        self.v = tk.StringVar(None, 'SGD')
        for name in optimizers:
            tk.Radiobutton(self.master, text=name, variable=self.v, value=name).pack()


        tk.Label(self.master, text='Learning Rate:').pack()
        self.lr = tk.Entry(self.master)
        self.lr.pack()

        self.button = tk.Button(self.master, text='Ok', command = self.get_opt, width=20)
        self.button.pack()

    def get_opt(self):
        lr = self.lr.get()
        opt = self.OPS[self.v.get()]
        GLOBAL_OPTIMIZER[0] = opt(learning_rate=float(lr))
        self.controller.OPT.configure(text=self.v.get(), bg='green')
        self.master.destroy()




class Build:
    def __init__(self, master):
        self.master = master
#        self.master.geometry("500x500")
        self.master.title('Model Build')
        self.frame = tk.Frame(self.master)
        tk.Button(self.frame, text = 'Input',                   width=40, height=5, command = lambda:input_window(tk.Toplevel(self.master))).grid(row=0, column=0)
        tk.Button(self.frame, text = 'Insert Primitives',       width=40, height=5, command = lambda:Primitives  (tk.Toplevel(self.master))).grid(row=1, column=0)
        tk.Button(self.frame, text = 'Insert Blocks',           width=40, height=5, command = lambda:Blocks      (tk.Toplevel(self.master))).grid(row=2, column=0)
        tk.Button(self.frame, text = 'View Model',              width=40, height=5, command = self.view_model).grid(row=3, column=0)
        tk.Button(self.frame, text = 'Remove Previous Layer',   width=40, height=5, command = self.erase).grid(row=4, column=0)
        tk.Button(self.frame, text = 'Exit',                    width=40, height=5, command = lambda:quit(self.master)).grid(row=5, column=0)
        self.frame.pack()


    def view_model(self):
        new_window = tk.Toplevel(self.master)
        ViewModel(new_window)


    def erase(self):
        new_window = tk.Toplevel(self.master)
        new_window.title('Erase')
        frame = tk.Frame(new_window)
        if GLOBAL_STRUCTURE:
            item = GLOBAL_STRUCTURE.pop(-1)
            tk.Label(frame, text=f'Layer: \n{item}\nDeleted Successfully!', width=40, height=5).pack()
        else:
            tk.Label(frame, text='No Layers to delete!', width=20, height=5).pack()
        frame.pack()
        new_window.after(1000, new_window.destroy)





class ViewModel:
    def __init__(self, master):
        self.master = master
        self.master.title('Model')
        self.frame = tk.Frame(self.master)

        model = assemble()
        if model:
            temp = []
            model.summary(print_fn=lambda x: temp.append(x))
            short_model_summary = "\n".join(temp)
            #self.master.after(1000)
            tk.Label(self.frame, text=short_model_summary, width=50, height=50, anchor='n').pack()
            tk.Button(self.frame, text = 'Close', width=40, height=5, command = lambda:quit(self.master)).pack()
        else:
            tk.Label(self.frame, text='No model to display!', width=40, height=5).pack()
            self.master.after(1000, self.master.destroy)

        self.frame.pack()

def quit(master):
    current = tk.Toplevel(master)
    current.title('Quit')
    frame = tk.Frame(current)
    tk.Label(frame, text='Are you sure?', width=20, height=3).grid(row=0, column=0, columnspan=2)
    tk.Button(frame, text='Yes', width=10, height=3, command=lambda:master.destroy()).grid(row=1, column=0)
    tk.Button(frame, text='No',  width=10, height=3, command=lambda:current.destroy()).grid(row=1, column=1)
    frame.pack()
    

class Blocks:
    def __init__(self, master):
        self.master = master
        self.master.title('Blocks')
        self.frame = tk.Frame(self.master)

        tk.Button(self.frame, text='Inception',             command=self.inception).pack()
        tk.Button(self.frame, text='Residual',              command=self.residual).pack()
        tk.Button(self.frame, text='VGG',                   command=self.VGG).pack()
        tk.Button(self.frame, text='Squeeze and Excite',    command=self.squeezeexcite).pack()
        tk.Button(self.frame, text='Inverted Residual',     command=self.inverted_residual).pack()
        self.frame.pack()


    def inception(self):
        #parameterized by channels
        GLOBAL_STRUCTURE.append(Inception())

    def residual(self):
        #parameterized by channels
        GLOBAL_STRUCTURE.append(Residual())
        print('residual added')

    def VGG(self):
        #parameterized by channels and n-number multiplier of the block
        GLOBAL_STRUCTURE.append(VGG())
        print('VGG added')

    def squeezeexcite(self):
        #parameterized by ratio
        GLOBAL_STRUCTURE.append(SqueezeExcite())
        print('sne added')

    def inverted_residual(self):
        GLOBAL_STRUCTURE.append(InvertedResidual())
        print('inverted residual added')


class Primitives:
    def __init__(self, master):
        self.master = master
        self.master.title('Primitive Operations')
        self.frame = tk.Frame(self.master)


        tk.Button(self.frame, text='Dense',       command=lambda: dense(tk.Toplevel(self.master))).pack()
        tk.Button(self.frame, text='Conv1D',      command=lambda: conv_window(tk.Toplevel(self.master), 'Conv1D')).pack()
        tk.Button(self.frame, text='Conv2D',      command=lambda: conv_window(tk.Toplevel(self.master), 'Conv2D')).pack()
        tk.Button(self.frame, text='Conv3D',      command=lambda: conv_window(tk.Toplevel(self.master), 'Conv3D')).pack()
        tk.Button(self.frame, text='SepConv1D',   command=lambda: conv_window(tk.Toplevel(self.master), 'SepConv1D')).pack()
        tk.Button(self.frame, text='SepConv2D',   command=lambda: conv_window(tk.Toplevel(self.master), 'SepConv2D')).pack()
#        self.button5= tk.Button(self.frame, text='DepConv1D',   command=self.depconv1d)
        tk.Button(self.frame, text='DepConv2D',   command=lambda: conv_window(tk.Toplevel(self.master), 'DepConv2D')).pack()
        tk.Button(self.frame, text='Conv2DTran',  command=lambda: conv_window(tk.Toplevel(self.master), 'Conv2DTrans')).pack()
        tk.Button(self.frame, text='Conv3DTran',  command=lambda: conv_window(tk.Toplevel(self.master), 'Conv3DTrans')).pack()

        self.frame.pack()


class dense:
    def __init__(self, master):
        self.master = master
        self.master.title("Dense")


        tk.Label(self.master, text='Nodes:').pack()
        self.nodes = tk.Entry(self.master)
        self.nodes.pack()


        activations = ['ReLU', 'Sigmoid', 'TanH', 'Hard Sigmoid', 'Linear']
        self.ACTS   = {'ReLU': relu, 'Sigmoid': sigmoid, 'TanH': tanh, 'Hard Sigmoid': hard_sigmoid, 'Linear': linear}

        self.v = tk.StringVar(None, 'Linear')
        for name in activations:
            tk.Radiobutton(self.master, text=name, variable=self.v, value=name).pack()

        self.button = tk.Button(self.master, text='Ok', command = self.get_act, width=20)
        self.button.pack()

    def get_act(self):
        if not GLOBAL_STRUCTURE:
            GLOBAL_STRUCTURE.append(Dense(int(self.nodes.get()), input_shape=INPUT_SHAPE[0]))
        else:
            GLOBAL_STRUCTURE.append(Dense(int(self.nodes.get())))

        tk.Label(self.master, text='Input Successfully Saved').pack()
        self.master.after(1500, self.master.destroy)


class conv_window:
    def __init__(self, master, type):
        self.master = master
        self.master.title(type)
        self.temp = tk.Frame(self.master)
        self.type = type
        if type != 'DepConv2D':
            tk.Label(self.temp, text='Number of filters:').pack()
            self.filter_input = tk.Entry(self.temp)
            self.filter_input.pack()

        tk.Label(self.temp, text='Kernel size:').pack()
        self.kernel_input = tk.Entry(self.temp)
        self.kernel_input.pack()

        tk.Label(self.temp, text='Stride size:').pack()
        self.stride_input = tk.Entry(self.temp)
        self.stride_input.pack()

        tk.Label(self.temp, text='Stride type:').pack()
        if type == 'DepConv2d':
            self.depth_input = tk.Label(self.temp, text='Depth Multiplier')
            self.depth_input.pack()
            tk.Entry(self.temp).pack()

        padding_options = ['same', 'valid']
        self.padding_value = tk.StringVar(self.temp)
        self.padding_value.set(padding_options[0])

        tk.OptionMenu(self.temp, self.padding_value, *padding_options).pack()


        self.warning_text   = tk.StringVar()
        self.warning_label  = tk.Label(self.temp, textvariable=self.warning_text)
        self.warning_label.pack()

        self.button = tk.Button(self.temp, text='Ok', command = self.get_conv)
        self.button.pack()

        self.temp.pack()

    def get_conv(self):
        if self.type == 'DepConv2D':
            if self.depth_input.get().isdigit():
                filter = int(self.depth_input.get())
            else:
                self.warning_text.set('Filter Size Invalid')
                return
        else:
            if self.filter_input.get().isdigit():
                filter = int(self.filter_input.get())
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
        commands = {
        'Conv1D': Conv1D,
        'Conv2D': Conv2D,
        'Conv3D': Conv3D,
        'SepConv1D': SeparableConv1D,
        'SepConv2D': SeparableConv1D,
        'Conv2DTrans': Conv2DTranspose,
        'Conv3DTrans': Conv3DTranspose
        }

        #Conv2D(filters=filter, kernel_size=kernel, strides=stride, padding=padding)
        print(INPUT_SHAPE)
        if not GLOBAL_STRUCTURE:
            op = commands[self.type](filters=filter, kernel_size=kernel, strides=stride, padding=padding, input_shape=INPUT_SHAPE[0])
        else:
            op = commands[self.type](filters=filter, kernel_size=kernel, strides=stride, padding=padding)
        GLOBAL_STRUCTURE.append(op)
        tk.Label(self.temp, text='Input Successfully Saved').pack()
        self.temp.after(1500, self.master.destroy)

#keras.layers.Conv1D(filters, kernel_size, strides=1, padding='valid', data_format='channels_last', dilation_rate=1, activation=None, use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)


class input_window:
    def __init__(self, master):
        self.master = master
        self.temp = tk.Frame(self.master)
        self.label1 = tk.Label(self.temp, text='Dimensions:', width=30, height=3, font='24')
        #self.label2 = tk.Label(self.temp, text='Input Channels:')
        self.input1 = tk.Entry(self.temp, font='Calibri 24')
        #self.input2 = tk.Entry(self.temp)
        self.button = tk.Button(self.temp, text='Ok',width=10, height=3, command = self.get_input)
        self.label1.pack()
        self.input1.pack()
        #self.label2.pack()
        #self.input2.pack()
        self.button.pack()
        self.temp.pack()
        master.bind('<Return>', lambda event: self.get_input())

    def get_input(self):
        i = tuple([int(x.strip()) for x in self.input1.get().split(',')])
        INPUT_SHAPE.append(i)
        self.button.destroy()
        self.label1.destroy()
        self.input1.destroy()
        tk.Label(self.temp, text='Dimensions: {}'.format(i)).pack()
        tk.Label(self.temp, text='input successfully saved').pack()
        self.temp.after(1500, self.master.destroy)
        #self.master.destroy()


def assemble():
    model = None
    if GLOBAL_STRUCTURE:
        model = Sequential()
        for item in GLOBAL_STRUCTURE:
            model.add(item)
        #print(model.summary())
    else:
        print('nothing to display')

    GLOBAL_MODEL[0] = model

def main():
    root = tk.Tk()
    face = display(root)
    root.mainloop()



    print(GLOBAL_STRUCTURE)
    print(GLOBAL_MODEL)
    #create sequential model with GUI
#    model = assemble()
#    model.summary()

if __name__ == '__main__':
    main()
