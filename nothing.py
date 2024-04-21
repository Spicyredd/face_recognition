import tkinter as tk
import pandas as pd

def read_csv():
    # Read the CSV file
    df = pd.read_csv('data.csv')  # Replace 'data.csv' with your CSV file path
    # Update the label text with the first row of the DataFrame
    label.config(text=df.iloc[0].to_string())

    # Schedule the read_csv function to be called again after 1000 milliseconds (1 second)
    root.after(1000, read_csv)

# Create the Tkinter window
root = tk.Tk()
root.title("CSV Reader")

# Create a label to display the data
label = tk.Label(root, text="")
label.pack()

# Start reading the CSV file
read_csv()

# Run the Tkinter event loop
root.mainloop()
