import tkinter as tk


root= tk.Tk()


gatename = ""
canvas1 = tk.Canvas(root, width = 400, height = 300)
canvas1.pack()
label1 = tk.Label(root, text='Specify Gate Name')
label1.config(font=('helvetica', 14))
canvas1.create_window(200, 25, window=label1)
entry1 = tk.Entry (root) 
canvas1.create_window(200, 140, window=entry1)

def getName():  
    global gatename
    gatename = entry1.get()
    root.destroy()
    button1 = tk.Button(text='Set Gate Name', command=getName)
    canvas1.create_window(200, 180, window=button1)

root.mainloop()
