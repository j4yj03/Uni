import tkinter as tk
from tkinter import scrolledtext
from tkinter import simpledialog

window = tk.Tk()
# the input dialog
user = simpledialog.askstring(title="Chat-Name",
                                  prompt="Gib deinen Chat-Namen ein:")

window.title("Simple Chat-Program - " + 'Nutzer: ' + user)

label1 = tk.Label(window, text="Deine Nachricht", font='Helvetica 18 bold')
label2 = tk.Label(window, text="Chat-Verlauf", font='Helvetica 18 bold')
label3 = tk.Label(window, text="Gib hier deine Nachricht ein:")
entry1 = tk.Entry(window, width=50)
text1 = scrolledtext.ScrolledText(window, width=70, bg = "gray", wrap='word')

def b1CallBack():
    tk.messagebox.showinfo( "Hinweis", "Chat-Nachricht wurde versendet!")
    text1.insert("end", entry1.get()+'\n') # kann später entfallen, da dieses Feld durch 
                                      # eingehende MQTT-Nachrichten gefüllt wird
   #client.publish()

b1 = tk.Button(window, text ="Nachricht absenden", command = b1CallBack)

# Layout der Bildschirmelemente als Grid
label1.grid(row=0, column=0)
label2.grid(row=0, column=1)
label3.grid(row=1, column=0, sticky='SW')
entry1.grid(row=2, column=0, sticky='N')
b1.grid(row=3, column=0, sticky='N')
text1.grid(row=1, column=1, rowspan=3)

#Main MQTT-Code
window.mainloop()