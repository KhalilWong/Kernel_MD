import tkinter
#
root = tkinter.Tk()
#
Title = tkinter.Label(root, text='AWSL AWSL AWSL!')
Title.grid(row = 0, column = 0, columnspan = 3)
#
Label0 = tkinter.Label(root, text='A')
Label0.grid(row = 1, column = 0)
#
Entry0 = tkinter.Entry(root)
Entry0.grid(row = 1, column = 1)
Entry0.insert(0, '1')
#
Label1 = tkinter.Label(root, text='W')
Label1.grid(row = 2, column = 0)
#
Entry1 = tkinter.Entry(root)
Entry1.grid(row = 2, column = 1)
Entry1.insert(0, '23')
#
Label2 = tkinter.Label(root, text='S')
Label2.grid(row = 3, column = 0)
#
Entry2 = tkinter.Entry(root)
Entry2.grid(row = 3, column = 1)
Entry2.insert(0, '19')
#
Label3 = tkinter.Label(root, text='L')
Label3.grid(row = 4, column = 0)
#
Entry3 = tkinter.Entry(root)
Entry3.grid(row = 4, column = 1)
Entry3.insert(0, '12')
#
Button0 = tkinter.Button(root, text='Who?')
Button0.grid(row = 1, column = 2, rowspan = 5)
#
chkBtnVar = tkinter.IntVar()
chkBtn = tkinter.Checkbutton(root, text='Resurrection可能吗!')
chkBtn.grid(row = 5, column = 0, columnspan = 2)
#
root.mainloop()
