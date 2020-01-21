import tkinter as tk
import time
from .engine.input_layer import Input
from .layers.convolutional import Conv1D, Conv2D, Conv3D, SeparableConv1D, SeparableConv2D, DepthwiseConv2D, Conv2DTranspose, Conv3DTranspose

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

        tk.Button(self.frame, text='Input',       command=self.input).pack()
        tk.Button(self.frame, text='Conv1D',      command=self.conv1d).pack()
        tk.Button(self.frame, text='Conv2D',      command=self.conv2d).pack()
        tk.Button(self.frame, text='Conv3D',      command=self.conv3d).pack()
        tk.Button(self.frame, text='SepConv1D',   command=self.sepconv1d).pack()
        tk.Button(self.frame, text='SepConv2D',   command=self.sepconv2d).pack()
#        self.button5= tk.Button(self.frame, text='DepConv1D',   command=self.depconv1d)
        tk.Button(self.frame, text='DepConv2D',   command=self.depconv2d).pack()
        tk.Button(self.frame, text='Conv2DTran',  command=self.conv2dtrans).pack()
        tk.Button(self.frame, text='Conv3DTran',  command=self.conv3dtrans).pack()

#        self.button10=tk.Button(self.frame, text='
#        self.button.pack()
#        self.button1.pack()
        self.frame.pack()


    def input(self):
        new_window = tk.Toplevel(self.master)
        temp = input_window(new_window)

    def conv1d(self):
        new_window = tk.Toplevel(self.master)
        temp = conv_window(new_window, 'Conv1D')

    def conv2d(self):
        #need to add activation to the list of hyperparams
        new_window = tk.Toplevel(self.master)
        temp = conv_window(new_window, 'Conv2D')

    def conv3d(self):
        new_window = tk.Toplevel(self.master)
        temp = conv_window(new_window, 'Conv3D')

    def sepconv1d(self):
        new_window = tk.Toplevel(self.master)
        temp = conv_window(new_window, 'SepConv1D')

    def sepconv2d(self):
        new_window = tk.Toplevel(self.master)
        temp = conv_window(new_window, 'SepConv2D')

    def depconv2d(self):
        new_window = tk.Toplevel(self.master)
        temp = conv_window(new_window, 'DepConv2D')

    def conv2dtrans(self):
        new_window = tk.Toplevel(self.master)
        temp = conv_window(new_window, 'Conv2DTrans')

    def conv3dtrans(self):
        new_window = tk.Toplevel(self.master)
        temp = conv_window(new_window, 'Conv3DTrans')



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
        'Conv1D': Conv1D(filters=filter, kernel_size=kernel, strides=stride, padding=padding),
        'Conv2D': Conv2D(filters=filter, kernel_size=kernel, strides=stride, padding=padding),
        'Conv3D': Conv3D(filters=filter, kernel_size=kernel, strides=stride, padding=padding),
        'SepConv1D': SeparableConv1D(filters=filter, kernel_size=kernel, strides=stride, padding=padding),
        'SepConv2D': SeparableConv1D(filters=filter, kernel_size=kernel, strides=stride, padding=padding),
        'Conv2DTrans': Conv2DTranspose(filters=filter, kernel_size=kernel, strides=stride, padding=padding),
        'Conv3DTrans': Conv3DTranspose(filters=filter, kernel_size=kernel, strides=stride, padding=padding)
        }

        op = commands[self.type] #Conv2D(filters=filter, kernel_size=kernel, strides=stride, padding=padding)
        GLOBAL_STRUCTURE.append(op)
        tk.Label(self.temp, text='Input Successfully Saved').pack()
        self.temp.after(1500, self.master.destroy)







#keras.layers.Conv1D(filters, kernel_size, strides=1, padding='valid', data_format='channels_last', dilation_rate=1, activation=None, use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)




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

    #create sequential model with GUI
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
