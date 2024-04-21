import tkinter as tk
from tkinter import ttk
from live_detector import *
import csv
from threading import Thread

def exec():
    display_csv_data()
    video_thread = Thread(target=recognize_faces_in_video)
    video_thread.daemon = True
    video_thread.start()

def display_csv_data():
    file_path = 'students.csv'
    try:
        with open(file_path, 'r', newline='') as file:
            csv_reader = csv.reader(file)
            header = next(csv_reader)  # Read the header row
            tree.delete(*tree.get_children())  # Clear the current data

            tree["columns"] = header
            for col in header:
                tree.heading(col, text=col)
                tree.column(col, width=100)

            for row in csv_reader:
                tree.insert("", "end", values=row)

            status_label.config(text=f"CSV file loaded: {file_path}")
            root.after(1000,display_csv_data)

    except Exception as e:
        status_label.config(text=f"Error: {str(e)}")

root = tk.Tk()
root.title("CSV File Viewer")

open_button = tk.Button(root, text="Start", command=exec)
open_button.pack(padx=20, pady=10)


# display_csv_data()

tree = ttk.Treeview(root, show="headings")
tree.pack(padx=20, pady=20, fill="both", expand=True)

status_label = tk.Label(root, text="", padx=20, pady=10)
status_label.pack()


root.mainloop()