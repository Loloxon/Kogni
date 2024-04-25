import tkinter as tk
from tkinter import filedialog

import numpy as np
from PIL import Image, ImageTk
from tensorflow.keras.models import load_model

# Load the pre-trained model
position_model = load_model("models/position_model.h5")
type_model = load_model("models/type_model.h5")
healthy_model = load_model("models/healthy_model.h5")

# Define your image dimensions
image_height = 640
image_width = 640

label_position_dict = {0: "palca", 1: "dłoni", 2: "nadgarstka", 3: "wielokrotne"}
label_type_dict = {0: "Złamanie", 1: "Przesunięcie", 2: "Inne uszkodzenie"}

label_position_dict_info = {0: "Uszkodzenie występuje w obszarze kości palców.",
                            1: "Uszkodzenie występuje na obszarze kości dłoni.",
                            2: "Uszkodzenie występuje w okolicy nadgarstka lub stawu dłoni.",
                            3: "Uszkodzenie wielu kości lub wielu obszarów."}
label_type_dict_info = {
    0: "Wykryto złamanie. Należy nie ruszać uszkodzonego obszaru oraz jak najszybciej skontaktować się z lekarzem.\n"
       " Możliwe że konieczne będzie założenie gipsu.",
    1: "Wykryto przesunięcie. W przypadku drobnego uszkodzenia zalecana próba nastawienia,\n"
       " w przypadku bardziej rozległego konieczna operacja.",
    2: "Wykryto uszkodzenie inne od złamania lub przesunięcia. Prawdopodobnie jest to stłuczenie,\n"
       " lecz precyzyjna ocena wymaga bezpośredniej analizy lekarskiej."}

additional_info_dict = {}

for i in range(4):
    for j in range(3):
        key = (i, j)
        value = (f"{label_position_dict_info[i]}\n"
                 f"{label_type_dict_info[j]}")
        additional_info_dict[key] = value

# for key, value in additional_info_dict.items():
#     print(key, ":", value)


# Function to perform prediction
def predict_label(image_path):
    image = Image.open(image_path)
    image = image.resize((image_height, image_width))
    image = np.array(image)
    # image = np.array(image) / 255.0  # Normalize the image
    image = np.expand_dims(image, axis=0)  # Add batch dimension

    # Perform prediction
    # print(image)
    predictions_pos = position_model.predict(image)
    predictions_type = type_model.predict(image)
    predictions_healthy = healthy_model.predict(image)
    # print(predictions_pos)
    # print(predictions_type)
    # print(predictions_healthy)
    label_pos_index = np.argmax(predictions_pos)
    label_type_index = np.argmax(predictions_type)
    is_healthy = np.argmax(predictions_healthy)
    predicted_label_pos = label_position_dict[label_pos_index]
    predicted_label_type = label_type_dict[label_type_index]

    # additional_info = (f"Prediction position probabilities: {predictions_pos.squeeze()}\n"
    #                    f"Prediction type probabilities: {predictions_type.squeeze()}")
    additional_info = additional_info_dict[(label_pos_index, label_type_index)]

    return predicted_label_pos, predicted_label_type, additional_info, Image.open(image_path), is_healthy


# Function to handle file upload
def upload_file():
    file_path = filedialog.askopenfilename()
    label_pos, label_type, additional_info, image, is_healthy = predict_label(file_path)
    image.thumbnail((400, 400))  # Resizing image for display
    imgtk = ImageTk.PhotoImage(image)
    image_label.config(image=imgtk)
    image_label.image = imgtk
    # result_label.config(text=additional_info)
    if is_healthy:
        result_label.config(text="Brak wykrytego uszkodzenia")
        additional_info_label.config(text="")
    else:
        result_label.config(text="Wykryto: " + label_type + " " + label_pos)
        additional_info_label.config(text=additional_info)


# Create the main window
root = tk.Tk()
root.title("Image Classifier")
root.geometry("800x600")

# Create a button to upload the file
upload_button = tk.Button(root, text="Upload Image", command=upload_file)
upload_button.pack(pady=10)

# Create a label to display the image
image_label = tk.Label(root)
image_label.pack(pady=10)

# Create a label to display the result
result_label = tk.Label(root, text="")
result_label.pack(pady=10)

additional_info_label = tk.Label(root, text="")
additional_info_label.pack(pady=10)
# Run the application
root.mainloop()
