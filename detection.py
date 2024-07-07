# Predict animal and give warning
# Note: This code is recommended to be run on a local machine due to hardware requirements and access to files (sound file)
import cv2
import torch
import numpy as np
import tkinter as tk
import time
import pygame
class_names = ["Bear", "Bull", "Cheetah", "Crocodile", "Elephant", "Jaguar", "Leopard", "Lion", "Rhinoceros", "Tiger"]
camera=cv2.VideoCapture(0)
pop_up_shown = False
pop_up_last_time = 0

def show_alert(animal_class):
    play_alert_sound(animal_class)  # Play the alert sound
    root = tk.Tk()
    root.title("Animal Alert")
    alert_label = tk.Label(root, text=f"An {animal_class} has been detected!", font=("Helvetica", 16))
    alert_label.pack(padx=20, pady=20)
    root.after(3000, root.destroy)  # Close the pop-up after 3 seconds
    root.mainloop()

# def play_alert_sound():
#     pygame.mixer.init()
#     pygame.mixer.music.load(r'D:\mini_project\alarm.wav')  # Provide the path to your alert sound file
#     pygame.mixer.music.play()

def play_alert_sound(animal_class):
    pygame.mixer.init()
    alert_sound_path = r'D:\mini_project\alarm.wav'  # Path to your alert sound file
    animal_sound_path = r'D:\mini_project\ambul1.wav'  # Path to your animal sound file
    lion_sound_path = r'D:\mini_project\lion.wav'  # Path to your animal sound file
    bear_sound_path = r'D:\mini_project\bear.wav'  # Path to your animal sound file
    dangeranimal_sound_path = r'D:\mini_project\dangeranimal.wav'  # Path to your animal sound file
    thunder_sound_path = r'D:\mini_project\thunder.wav'  # Path to your animal sound file
    # Load and play the alert sound
    pygame.mixer.music.load(alert_sound_path)
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():
        pygame.time.Clock().tick(10)  # Adjust the tick rate as needed
    print(animal_class.iloc[0])
    print("%"*50)
    if(animal_class.iloc[0]=="Bear"):
        # Load and play the animal sound
        pygame.mixer.music.load(thunder_sound_path)
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(10)  # Adjust the tick rate as needed
    
    if(animal_class.iloc[0]=="Bull"):
        # Load and play the animal sound
        pygame.mixer.music.load(bear_sound_path)
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(10)  # Adjust the tick rate as needed
    if(animal_class.iloc[0]=="Cheetah"):
        # Load and play the animal sound
        pygame.mixer.music.load(dangeranimal_sound_path)
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(10)  # Adjust the tick rate as needed
    if(animal_class.iloc[0]=="Crocodile"):
        # Load and play the animal sound
        pygame.mixer.music.load(dangeranimal_sound_path)
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(10)  # Adjust the tick rate as needed
    if(animal_class.iloc[0]=="Elephant"):
        # Load and play the animal sound
        pygame.mixer.music.load(thunder_sound_path)
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(10)  # Adjust the tick rate as needed
    if(animal_class.iloc[0]=="Jaguar"):
        # Load and play the animal sound
        pygame.mixer.music.load(thunder_sound_path)
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(10)  # Adjust the tick rate as needed
    if(animal_class.iloc[0]=="Leopard"):
        # Load and play the animal sound
        pygame.mixer.music.load(bear_sound_path)
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(10)  # Adjust the tick rate as needed
    if(animal_class.iloc[0]=="Lion"):
        # Load and play the animal sound
        pygame.mixer.music.load(dangeranimal_sound_path)
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(10)  # Adjust the tick rate as needed
    if(animal_class.iloc[0]=="Rhinoceros"):
        # Load and play the animal sound
        pygame.mixer.music.load(lion_sound_path)
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(10)  # Adjust the tick rate as needed
    if(animal_class.iloc[0]=="Tiger"):
        # Load and play the animal sound
        pygame.mixer.music.load(thunder_sound_path)
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(10)  # Adjust the tick rate as needed
    
    
    



def detect_objects(camera, class_names, confidence_threshold, model_path):
    global pop_up_shown, pop_up_last_time  # Declare pop_up_shown and pop_up_last_time as global variables

    model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path)
    model.conf = confidence_threshold

    while True:
        ret, frame = camera.read()

        with torch.no_grad():
            results = model(frame)

        predictions = results.pandas().xyxy[0]
        print(predictions)
        predictions = predictions[predictions['confidence'] >= confidence_threshold]
        labels = predictions['class'].astype(int).tolist()
        boxes = predictions[['xmin', 'ymin', 'xmax', 'ymax']].values.tolist()

        for label, box in zip(labels, boxes):
            if len(box) != 4:
                print(f"Invalid box: {box}")
                continue
            x_min, y_min, x_max, y_max = map(int, box)
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            cv2.putText(frame, str(predictions["name"]), (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            if not pop_up_shown or time.time() - pop_up_last_time >= 0.5:
                animal_class = class_names[label]
                show_alert(predictions["name"])
                pop_up_shown = True
                pop_up_last_time = time.time()

        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    camera.release()
    cv2.destroyAllWindows()

# Start the initial detection process
detect_objects(camera, class_names, confidence_threshold=0.5, model_path=r'D:\mini_project\best.pt')