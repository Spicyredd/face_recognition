from tkinter import *
from live_detector import *
from detector import *
class MainGui:
    def __init__(self, title, geometry):
        self.root = Tk()
        self.root.title(title)
        self.root.geometry(geometry)
        btn = Button(self.root, text = 'This button ', bd = '5', command = recognize_faces)
        btn.pack(side = 'top')
        self.root.mainloop()
        
gui = MainGui('Mygui', '600x600')

# master = Tk()
# w = Canvas(master, width=40, height=40)
# w.pack()
# canvas_height = 20
# canvas_width = 200
# y = int(canvas_height / 2)
# w.create_line(0, y, canvas_width, y)
# mainloop()