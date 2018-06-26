
print("**********************\n*******Tests***********\n******************************")
print ("Python works")

try:
    import cv2
except ImportError:
    print("Open Cv not installed")

print ("OpenCV Works: " + cv2.__version__)

try:
    from Tkinter import *
    root = Tk()
    w = Label(root, text="Tkinter Works")
    w.pack()
    root.mainloop()
except ImportError:
    try:
        from tkinter import *
        root = Tk()
        w = Label(root, text="Tkinter Works")
        w.pack()
        root.mainloop()
    except ImportError:
        print("Tkinter not installed")

print("Tkinter works")
