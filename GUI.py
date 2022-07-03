import tkinter as tk
from tkinter import NW
from tkinter.filedialog import asksaveasfilename, askopenfilename
from tkinter.font import Font
from BackEnd import setVolume
from BackEnd import Speak

window = tk.Tk()
window.title("Text To Speech")
window.geometry("700x550")


# open file explorer and choose a txt file
def open_file():
    filepath = askopenfilename(
        filetypes=[("Text Files", "*.txt"), ("All Files", "*.*")]
    )
    if not filepath:
        return
    text_area.delete("1.0", tk.END)
    with open(filepath, mode="r", encoding="utf-8") as input_file:
        text = input_file.read()
        text_area.insert(tk.END, text)
    window.title(f"Text To Speech - {filepath}")


# save all the changes that were made in the text area of the gui
def save_file():
    filepath = asksaveasfilename(
        defaultextension=".txt",
        filetypes=[("Text Files", "*.txt"), ("All Files", "*.*")],
    )
    if not filepath:
        return
    with open(filepath, mode="w", encoding="utf-8") as output_file:
        text = text_area.get("1.0", tk.END)
        output_file.write(text)
    window.title(f"Text To Speech- {filepath}")


#
def increase():
    value = int(lbl_value["text"])
    if value >= 10:
        return

    lbl_value["text"] = f"{value + 1}"
    norm_value = (float(value) + 1) / 10
    setVolume(norm_value)


def decrease():
    value = int(lbl_value["text"])
    if value <= 0:
        return
    lbl_value["text"] = f"{value - 1}"
    norm_value = (float(value) - 1) / 10
    setVolume(norm_value)


def clearText():
    text_area.delete("1.0", tk.END)


def loadtext():
    text = text_area.get("1.0", tk.END)
    Speak(text)


my_font = Font(
    family='Times',
    size=20,
    weight='bold',
    slant='roman',
)

text_font = Font(family='Times',
                 size=15,
                 slant='roman',
                 )

label = tk.Label(text="Enter or import your text here: ",
                 foreground="black",
                 background="Yellow",
                 font=my_font
                 )
label.place(anchor=NW)


text_area = tk.Text(font=text_font,
                    background="white",
                    foreground="black",
                    relief='groove')
text_area.place(x=20,
                y=50,
                width=390,
                height=480)


start_button = tk.Button(
    text="Play",
    bg="purple",
    fg="yellow",
    font=my_font,
    command=loadtext
)
start_button.place(x=480, y=70, width=130, height=100)


open_button = tk.Button(
    text="Open",
    bg="purple",
    fg="yellow",
    font=my_font,
    command=open_file
)
open_button.place(x=450, y=190)

save_button = tk.Button(
    text="Save",
    bg="purple",
    fg="yellow",
    font=my_font,
    command=save_file
)
save_button.place(x=560, y=190)


clear_button = tk.Button(
    text="Clear Text",
    bg="purple",
    fg="yellow",
    font=my_font,
    command=clearText

)
clear_button.place(x=470, y=260)

volume_label = tk.Label(text="Volume",
                        foreground="purple",
                        font=my_font
                        )

volume_label.place(x=500,
                   y=350,
                   )

frame = tk.Frame(window)
frame.place(x=490,
            y=400,
            width=150,
            height=220)


btn_decrease = tk.Button(frame, text="-", font=my_font,width=2, command=decrease)
btn_decrease.grid(row=0, column=0, sticky="nsew")

lbl_value = tk.Label(frame, text="5", font=my_font,background='#ADD8E6', width=2)
lbl_value.grid(row=0, column=1)


btn_increase = tk.Button(frame, text="+", font=my_font,width=2, command=increase)
btn_increase.grid(row=0, column=2, sticky="nsew")

window.mainloop()
