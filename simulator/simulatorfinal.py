import customtkinter
from PIL import Image
import cv2 as cv
from PIL import Image, ImageTk, ImageDraw
import os
import cv2 as cv
import numpy as np
import csv
import argparse
import itertools
import copy
import tkinter as tk

from collections import Counter
from collections import deque
from utils import CvFpsCalc
from model import KeyPointClassifier
from model import PointHistoryClassifier
from sklearn.linear_model import LinearRegression
import mediapipe as mp
import sys
import joblib

customtkinter.set_appearance_mode("light")



class App(customtkinter.CTk):
    width = 1400
    height = 700

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.title("SignLingu - Filipino Sign Language Translator")
        self.geometry(f"{self.width}x{self.height}")
        self.resizable(False, False)  
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=1)         
        self.rf_model = joblib.load('rf_model.joblib')
        args = self.get_args()

        self.cap_device = args.device
        self.cap_width = args.width
        self.cap_height = args.height

        self.use_static_image_mode = args.use_static_image_mode
        self.min_detection_confidence = args.min_detection_confidence
        self.min_tracking_confidence = args.min_tracking_confidence

        self.use_brect = True
        self.is_playing = False
        self.predicted_class_index = None

        # Camera preparation ###############################################################
        self.cap = cv.VideoCapture(self.cap_device)
        self.cap.set(cv.CAP_PROP_FRAME_WIDTH, self.cap_width)
        self.cap.set(cv.CAP_PROP_FRAME_HEIGHT, self.cap_height)

        # Model load #############################################################
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=self.use_static_image_mode,
            max_num_hands=1,
            min_detection_confidence=self.min_detection_confidence,
            min_tracking_confidence=self.min_tracking_confidence,
        )

        self.keypoint_classifier = KeyPointClassifier()
        self.point_history_classifier = PointHistoryClassifier()

        # Read labels ###########################################################
        with open('model/keypoint_classifier/keypoint_classifier_label.csv', encoding='utf-8-sig') as f:
            keypoint_classifier_labels = csv.reader(f)
            self.keypoint_classifier_labels = [row[0] for row in keypoint_classifier_labels]
        with open('model/point_history_classifier/point_history_classifier_label.csv', encoding='utf-8-sig') as f:
            point_history_classifier_labels = csv.reader(f)
            self.point_history_classifier_labels = [row[0] for row in point_history_classifier_labels]

        # Load the dataset
        self.X, self.y = self.load_dataset('model/keypoint_classifier/keypoint.csv')

        # FPS Measurement ########################################################
        self.cvFpsCalc = CvFpsCalc(buffer_len=10)

        # Coordinate history #################################################################
        self.history_length = 16
        self.point_history = deque(maxlen=self.history_length)

        # Finger gesture history ################################################
        self.finger_gesture_history = deque(maxlen=self.history_length)
        # load images with light and dark mode image
        image_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "test_images")
        self.logo_image = customtkinter.CTkImage(Image.open(os.path.join(image_path, "logjhjfc.png")), size=(250, 90))
        self.rfsl_image = customtkinter.CTkImage(Image.open(os.path.join(image_path, "icon1.png")), size=(20, 20))
        self.dfsl_image = customtkinter.CTkImage(Image.open(os.path.join(image_path, "icon2.png")), size=(20, 20))
        self.blog_image = customtkinter.CTkImage(Image.open(os.path.join(image_path, "icon3.png")), size=(20, 20))
        self.update_image = customtkinter.CTkImage(Image.open(os.path.join(image_path, "updates_icon.png")), size=(15, 15))
        self.aboutus_image = customtkinter.CTkImage(Image.open(os.path.join(image_path, "about_icon.png")), size=(15, 15))
        self.large_test_image = customtkinter.CTkImage(Image.open(os.path.join(image_path, "homepagelogo.PNG")), size=(480, 180))
        self.image_icon_image = customtkinter.CTkImage(Image.open(os.path.join(image_path, "image_icon_light.png")), size=(20, 20))
        self.homebuttonicon = customtkinter.CTkImage(Image.open(os.path.join(image_path, "icon1.png")), size=(20, 20))
        self.playicon = customtkinter.CTkImage(Image.open(os.path.join(image_path, "play_icon.png")), size=(20, 20))
        self.pauseicon = customtkinter.CTkImage(Image.open(os.path.join(image_path, "pause_icon.png")), size=(20, 20))
        self.detbg = customtkinter.CTkImage(Image.open(os.path.join(image_path, "background.png")), size=(800, 230))
        self.actbg = customtkinter.CTkImage(Image.open(os.path.join(image_path, "h1.jpg")), size=(150, 230))
        self.yt = customtkinter.CTkImage(Image.open(os.path.join(image_path, "Youtube_logo.png")), size=(180, 120))
        
        # create sidebar frame
        self.sidebar_frame = customtkinter.CTkFrame(self, corner_radius=0, fg_color="#fff4dc", width=300, height=500,)
        self.sidebar_frame.grid(row=0, column=0, sticky="nsew")
        self.sidebar_frame.grid_rowconfigure(4, weight=1)
        self.sidebar_frame_label = customtkinter.CTkLabel(self.sidebar_frame, text="", image=self.logo_image,
                                                             compound="left")
        self.sidebar_frame_label.grid(row=0, column=0, padx=20, pady=20)
        self.home_button = customtkinter.CTkButton(self.sidebar_frame, corner_radius=32, height=40, border_spacing=10, text="Sign Language Translation", font=("Arial Bold", 14),
                                                   fg_color="#f4a808", text_color=("white", "gray90"), hover_color=("#8a7d62", "gray30"),
                                                   image=self.rfsl_image, anchor="w", 
                                                   command=self.home_button_event)
        self.home_button.grid(row=1, column=0, pady=(16, 10),padx=20, sticky="nsew" )

        self.dfsl_button = customtkinter.CTkButton(self.sidebar_frame, corner_radius=32, height=40, border_spacing=10, text="Filipino Sign Language", font=("Arial Bold", 14),
                                                   fg_color="#8a7d62", text_color=("white", "gray90"), hover_color=("#f4a808", "gray30"),
                                                   image=self.dfsl_image, anchor="w", 
                                                   command=self.dfsl_button_event)
        self.dfsl_button.grid(row=2, column=0, pady=(16, 10), sticky="nsew", padx=20)

        self.blog_button = customtkinter.CTkButton(self.sidebar_frame, corner_radius=32, height=40, border_spacing=10, text="Learn More about FSL", font=("Arial Bold", 14),
                                                   fg_color="#8a7d62", text_color=("white", "gray90"), hover_color=("#f4a808", "gray30"),
                                                   image=self.blog_image, anchor="w", 
                                                   command=self.blog_button_event)
        self.blog_button.grid(row=3, column=0, pady=(16, 10), sticky="nsew", padx=20)

        self.update_button = customtkinter.CTkButton(self.sidebar_frame, corner_radius=32, height=40, border_spacing=10, text="Updates & FAQ’s", font=("Arial", 14),
                                                   fg_color="transparent", text_color=("#000000", "gray90"), hover_color=("#ffc64e", "gray30"),
                                                   image=self.update_image, anchor="w", 
                                                   command=self.update_button_event
                                                   )
        self.update_button.grid(row=4, column=0, pady=(260, 1),sticky="nsew", padx=20)

        self.aboutus_button = customtkinter.CTkButton(self.sidebar_frame, corner_radius=32, height=40, border_spacing=10, text="About Us", font=("Arial", 14),
                                                   fg_color="transparent", text_color=("#000000", "gray90"), hover_color=("#ffc64e", "gray30"),
                                                   image=self.aboutus_image, anchor="w", 
                                                   command=self.aboutus_button_event
                                                   )
        self.aboutus_button.grid(row=5, column=0, pady=(1, 15), sticky="nsew",  padx=20)

        #HOMEPAGE FRAME ----------------------------------------------------------------------------------------------------------------------------------
        #grey
        self.home_frame = customtkinter.CTkFrame(self, corner_radius=0, fg_color="#e3e3e3", width=1400, height=700)
        self.home_frame.grid_columnconfigure(0, weight=1)

        #yellow
        self.search_container = customtkinter.CTkFrame(self.home_frame, fg_color="#fff4dc", border_width=2, border_color="#FFF0D5")
        self.search_container.grid(row=0, column=0, sticky="nsew", pady=(25, 0), padx=20, columnspan=3, rowspan=3)

        #contents
        self.home_frame_large_image_label = customtkinter.CTkLabel(self.search_container, text="", image=self.large_test_image,  anchor="center")
        self.home_frame_large_image_label.grid(row=0, column=3, padx=250, pady=(170, 0))

        self.homeframetext = customtkinter.CTkLabel(self.search_container, text="Powered by Data-Driven Technology - your partner in Sign Language Communication", font=("Arial", 14), text_color="#71797E")
        self.homeframetext.grid(row=1, column=3, padx=295, pady=(5, 30), sticky="nsew")
       
        self.homeframe_button_1 = customtkinter.CTkButton(self.search_container, text="Start New Sign Translation",font=("Arial Bold", 14),
            text_color="#FFFFFF", fg_color="#f4a808", height=50, width=1, hover_color="#8a7d62", border_color="#F4A808", border_width=2, image=self.homebuttonicon,  corner_radius=32, command=self.rfsl_button_event)
        self.homeframe_button_1.grid(row=3, column=3, padx=295, pady=(0,190))
    
        #REALTIMEFSL PAGE ----------------------------------------------------------------------------------------------------------------------------------
        self.rfsl_frame = customtkinter.CTkFrame(self, corner_radius=0, fg_color="#e3e3e3", width=1400, height=700)
        self.rfsl_frame.grid_columnconfigure(0, weight=1)
        
        #yellow
        self.rfsl_container = customtkinter.CTkFrame(self.rfsl_frame, fg_color="#fff4dc", border_width=2, border_color="#FFF0D5")
        self.rfsl_container.grid(row=0, column=0, sticky="nsew", pady=(25, 0), padx=20, columnspan=3, rowspan=3)
       
        #contents
        self.camera_label = customtkinter.CTkLabel(self.rfsl_container, text="",  anchor="center")
        self.camera_label.grid(row=0, column=1, padx=295, pady=(50, 30), sticky="nesw")

        self.start_camera()

        self.play_pause_button = customtkinter.CTkButton(self.rfsl_container, text="", text_color="#FFFFFF", fg_color="#f4a808", height=50, width=1, hover_color="#8a7d62", border_color="#F4A808", border_width=3, image=self.playicon,  corner_radius=32)
        self.play_pause_button.grid(row=2, column=1, padx=(12,130), pady=(0,140))
        self.play_pause_button.bind("<Button-1>", self.toggle_play_pause)

        self.prediction_entry = customtkinter.CTkEntry(self.rfsl_container,  width=100, height=50, border_color="#f4a808",font=("Arial Bold", 25), corner_radius=32)
        self.prediction_entry.grid(row=2, column=1, padx=(130,12), pady=(0,140))
        
        #DETAILEDFSL PAGE ----------------------------------------------------------------------------------------------------------------------------------
        image_paths = [
            "Sign_languages/A/6_jpg.rf.84f2161e258ea8a9e1792833c8fb95e7.jpg",
            "Sign_languages/B/125_jpg.rf.11108cca7999bdb7ce7053e4d1dc4a63.jpg",
            "Sign_languages/C/213_jpg.rf.77b70db29f5121475b2b61f93fa6948d.jpg",
            "Sign_languages/D/338_jpg.rf.96dd3cf308eeae54e21f36fe00bfe009.jpg",
            "Sign_languages/E/466_jpg.rf.bf0cc6bb9c78ddd02bfef6e5841bdc71.jpg",
            "Sign_languages/F/579_jpg.rf.d468820cb5c85fed82dbe411080f7523.jpg",
            "Sign_languages/G/848_jpg.rf.f2946ce118da3b747dc7d8e90bf5fac0.jpg",
            "Sign_languages/H/931_jpg.rf.0b568e904af2f426cee6fc809f57a1e4.jpg",
            "Sign_languages/I/1030_jpg.rf.24a1aa0721a909f9348b72109cd8ff7f.jpg",
            "Sign_languages/J/1124_jpg.rf.048488c6d20fa6a04b2527dcce81c6ab.jpg",
            "Sign_languages/K/1209_jpg.rf.118a8efae3499a4eb8b054ea8b0c9b33.jpg",
            "Sign_languages/L/1344-Copy_jpg.rf.8938eeeb02f2f0790d6d3fd780673a3f.jpg",
            "Sign_languages/M/1456_jpg.rf.028a3a5980e8ee2dbbcc0e85995cf7bd.jpg",
            "Sign_languages/N/1568_jpg.rf.44ed158b5d6b6e93bfcdfe188c16611e.jpg",
            "Sign_languages/O/1738_jpg.rf.ea2dca1c1d3145a7277f5921c1369400.jpg",
            "Sign_languages/P/2640_jpg.rf.c273a409b7ba4f641e70fbbcefec0e53.jpg",
            "Sign_languages/Q/2043_jpg.rf.8b1e20fcde0ad856533325aa84fb21b1.jpg",
            "Sign_languages/R/2165_jpg.rf.6071fc79632214775d03523e37097d8c.jpg",
            "Sign_languages/S/2368_jpg.rf.625060f2571ee58fce0b4825a5d64c3c.jpg",
            "Sign_languages/T/2517_jpg.rf.b183c02134987c1b3fb673a47ade523f.jpg",
            "Sign_languages/U/2621_jpg.rf.68014229c5dcc71bb62cb527c66b9fc7.jpg",
            "Sign_languages/V/2781_jpg.rf.1e089c68316fff9d48dc5fb9afce547b.jpg",
            "Sign_languages/W/2914_jpg.rf.b4e5b861c983ac0694e7314b399a82c6.jpg",
            "Sign_languages/X/3053_jpg.rf.177c7f5f61ea26ba2ba3774de5268c00.jpg",
            "Sign_languages/Y/3229_jpg.rf.b98ec5a2fcc0c82bea68b7601566dc06.jpg",
            "Sign_languages/Z/3338_jpg.rf.6f97b2e4651f6976684b892b26884750.jpg",
        ]

        # Create a list to store CTkImage objects and their corresponding labels
        self.image_label_pairs = []
        
        self.dfsl_frame = customtkinter.CTkFrame(self, corner_radius=0, fg_color="#e3e3e3", width=1400, height=700)
        self.dfsl_frame.grid_columnconfigure(0, weight=1)

        self.dfsl_container = customtkinter.CTkFrame(self.dfsl_frame, height=50, fg_color="#fff4dc", border_width=2, border_color="#FFF0D5")
        self.dfsl_container.grid(row=0, column=0, sticky="nsew", pady=(25, 0), padx=20, columnspan=3, rowspan=3)

        customtkinter.CTkLabel(self.dfsl_container, text="Filipino Sign Language", font=("Arial Bold", 30),
                 text_color="#8a7d62").pack(pady=(45, 0), padx=45, anchor="w")
        
        self.dfsl_row_container = customtkinter.CTkFrame(self.dfsl_container, fg_color="#fff4dc")
        self.dfsl_row_container.pack(fill="x", expand=True, )
        
        self.dfsl_canvas = customtkinter.CTkCanvas(self.dfsl_row_container)
        self.dfsl_canvas.configure(bg="#fff4dc", height=660)  # Set the canvas background color here

        self.dfsl_scrollbar = customtkinter.CTkScrollbar(self.dfsl_row_container, command=self.dfsl_canvas.yview)

        self.dfsl_canvas.configure(yscrollcommand=self.dfsl_scrollbar.set)

        self.dfsl_scrollbar.pack(side="right", fill="y")
        self.dfsl_canvas.pack(side="left", fill="both", expand=True, padx=100, pady=(20, 30))
        
         # Create a frame to hold the contents of the Canvas
        frame = customtkinter.CTkFrame(self.dfsl_canvas, fg_color="#fff4dc")
        self.dfsl_canvas.create_window((0, 0), window=frame, anchor="nw")

        # Function to update the scroll region
        def on_frame_configure(event):
            self.dfsl_canvas.configure(scrollregion=self.dfsl_canvas.bbox("all"))

        frame.bind("<Configure>", on_frame_configure)

        # Load and resize the images
        for i, path in enumerate(image_paths):
            self.original_image = Image.open(path)
            self.resized_image = self.original_image.resize((200, 200))
            photo_image = ImageTk.PhotoImage(self.resized_image)
            label_text = chr(i + 65)  # Convert index to character (A, B, C, ...)
            self.image_label_pairs.append((photo_image, label_text))

        # Create a list to store the card frames
        self.card_frames = []

                # Create cards for each image and label pair
        for i, (photo_image, label_text) in enumerate(self.image_label_pairs):
            self.card_frame = customtkinter.CTkFrame(frame, width=250, height=300, corner_radius=18, fg_color="#FFFFFF")
            self.card_frame.grid(row=i // 3, column=i % 3, padx=10, pady=5)
            self.card_frame.pack_propagate(0)
            self.card_frames.append(self.card_frame)
            
            # Bind a click event to the card_frame
            self.card_frame.bind("<Button-1>", lambda event, photo_image=photo_image, label_text=label_text: self.show_dfsl_details_frame(photo_image, label_text))

            # Create a label to display the image
            self.image_label = customtkinter.CTkLabel(self.card_frame, image=photo_image, text="")
            self.image_label.pack(fill="both", expand=True)

            # Create a frame as a separator with a background color
            self.separator_frame = tk.Frame(self.card_frame, width=180, height=2, bg="#8a7d62")
            self.separator_frame.pack(padx=8, pady=(1, 5))

            # Create a label for the image name or any other information
            customtkinter.CTkLabel(self.card_frame, text=label_text, font=("Arial Bold", 25), text_color="#71797E").pack(pady=(8, 18), anchor="center")
        
        self.dfsl_details_frame = customtkinter.CTkFrame(self, corner_radius=0, fg_color="#e3e3e3", width=1400, height=700)
        self.dfsl_details_frame.grid_columnconfigure(0, weight=1)
        
        self.dfsl_details_container = customtkinter.CTkFrame(self.dfsl_details_frame, height=50, fg_color="#fff4dc", border_width=2, border_color="#FFF0D5")
        self.dfsl_details_container.grid(row=0, column=0, sticky="nsew", pady=(25, 0), padx=20, columnspan=3, rowspan=3)
        
        back_icon = Image.open("simulator/icons/arrow.png")
        # Create the play/pause button
        back_button = customtkinter.CTkButton(
            self.dfsl_details_container,
            text="",
            fg_color="#fff4dc",
            hover_color="#fff4dc",
            height=2,
            width=2,
            image=customtkinter.CTkImage(dark_image=back_icon, light_image=back_icon),
            command=self.show_dfsl_frame
        )
        back_button.pack(pady=(20, 0), padx=45, anchor="w")
        

        # Create a frame with white background color
        self.dfsl_details_content_frame = customtkinter.CTkFrame(self.dfsl_details_container, fg_color="#FFFFFF", width=500, height=400, corner_radius=18)
        self.dfsl_details_content_frame.pack(pady=(20, 180))
        self.dfsl_details_content_frame.pack_propagate(0)

        # Display the photo image
        self.dfsl_selected_image_label = customtkinter.CTkLabel(master=self.dfsl_details_content_frame, image=None, text="")
        self.dfsl_selected_image_label.pack(pady=(50, 10), padx=45, anchor="center")

        # Create a frame as a separator with a background color
        separator_frame = tk.Frame(master=self.dfsl_details_content_frame, width=300, height=2, bg="#8a7d62")
        separator_frame.pack(padx=8, pady=(50, 5))

        # Display the label text
        self.dfsl_label = customtkinter.CTkLabel(master=self.dfsl_details_content_frame, text="", font=("Arial Bold", 30), text_color="#8a7d62")
        self.dfsl_label.pack(pady=(45, 20), anchor="center")

        #BLOGFSL PAGE ----------------------------------------------------------------------------------------------------------------------------------
        self.blog_frame = customtkinter.CTkFrame(self, corner_radius=0, fg_color="#e3e3e3", width=1400, height=700)
        self.blog_frame.grid_columnconfigure(0, weight=1)
        
        #yellow#fff4dc
        self.blog_container = customtkinter.CTkFrame(self.blog_frame, fg_color="#fff4dc", border_width=2, border_color="#FFF0D5")
        self.blog_container.grid(row=0, column=0, sticky="nsew", pady=(25, 25), padx=20, columnspan=3, rowspan=3)
       
        # create scrollable radiobutton frame
        self.scrollable_radiobutton_frame = customtkinter.CTkScrollableFrame(self.blog_container, width=500,  fg_color="#fff4dc",) 
                                                                       
        self.scrollable_radiobutton_frame.grid(row=1, column=0, padx=15, sticky="ns", pady=(25, 25))
        self.scrollable_radiobutton_frame.configure(width=1010, height=590)

        #contents                                                                       
        self.blog_container_bg_image = customtkinter.CTkLabel(self.scrollable_radiobutton_frame, text="", image=self.detbg, anchor="center")
        self.blog_container_bg_image.grid(row=0, column=0, padx=100, pady=(28,0), sticky="nsew")

        self.freference = customtkinter.CTkLabel(self.scrollable_radiobutton_frame, text="image from GRID Magazine", font=("Arial", 12),
                 text_color="grey" )
        self.freference.grid(row=1, column=0, padx=(0, 100), pady=0, sticky="e")

        self.heading = customtkinter.CTkLabel(self.scrollable_radiobutton_frame, text="Learn More About FSL", font=("Arial Bold", 20),
                 text_color="black" )
        self.heading.grid(row=2, column=0, padx=45, pady=20, sticky="w")

        self.subtitle1 = customtkinter.CTkLabel(self.scrollable_radiobutton_frame, text="Filipino Sign Language (FSL) is not merely a collection of gestures; it is a rich and distinct language with a history that predates foreign  \ninfluence by centuries. Originating as early as the 1590s on the island of Leyte, FSL has evolved into a complex linguistic system,  boasting \nunique structural features in its phonology, morphology, syntax, and discourse. Contrary to common misconceptions,  FSL is not \nsynonymous with American Sign Language (ASL); it possesses its own linguistic identity,  regional variations, and cultural nuances.", font=("Arial", 14),
                 text_color="black", justify="left" )
        self.subtitle1.grid(row=3, column=0, padx=45,  pady=(0, 5), sticky="w")

        self.subtitle2 = customtkinter.CTkLabel(self.scrollable_radiobutton_frame, text="FSL's significance within the Deaf community cannot be overstated. Recognized as one of over 100 distinct sign languages worldwide, FSL  \nenables Deaf Filipinos to communicate effectively, express their thoughts, and engage with their peers and society at large. Its usage \nextends beyond mere communication; it fosters cultural identity, pride, and solidarity among Deaf individuals across Luzon, Visayas, and \nMindanao.", font=("Arial", 14),
                 text_color="black", justify="left" )
        self.subtitle2.grid(row=4, column=0, padx=45,  pady=(10, 20), sticky="w")

        self.heading1 = customtkinter.CTkLabel(self.scrollable_radiobutton_frame, text="The Filipino Sign Language Act", font=("Arial Bold", 20),
                 text_color="black" )
        self.heading1.grid(row=5, column=0, padx=45, pady=(0,20), sticky="w")

        self.act_bg_image = customtkinter.CTkLabel(self.scrollable_radiobutton_frame, text="", image=self.actbg, anchor="center")
        self.act_bg_image.grid(row=6, column=0, padx=50, pady=(10,0), sticky="w")

        self.sreference = customtkinter.CTkLabel(self.scrollable_radiobutton_frame, text="image from freepik", font=("Arial", 12),
                 text_color="grey" )
        self.sreference.grid(row=7, column=0, padx=(50, 0), pady=0, sticky="nw")

        self.subtitle3 = customtkinter.CTkLabel(self.scrollable_radiobutton_frame, text="The Filipino Sign Language Act, passed in 2018, designates Filipino Sign Language (FSL) as the official sign \nlanguage of the Philippines.  This law ensures the rights of the deaf community by requiring its use in \ngovernment transactions and mandating the presence of qualified sign language interpreters. \n \nIn education, the act requires FSL to be used in teaching deaf learners and integrated as a separate \nsubject in their curriculum. Public institutions must provide information and services in \nFSL to ensure equal access for the deaf population. \n \nEmployment opportunities for the deaf are emphasized, with employers urged to provide \naccommodations for  accessibility and effective communication. The act also promotes \nthe preservation of FSL's cultural and linguistic  elements and advocates for public \nawareness campaigns to reduce stigma and foster inclusivity.", font=("Arial", 14),
        text_color="black", justify="right" )
        self.subtitle3.grid(row=6, column=0, padx=(0, 70), pady=(10, 0), sticky="ne")

        self.heading2 = customtkinter.CTkLabel(self.scrollable_radiobutton_frame, text="Practice FSL through Videos Online", font=("Arial Bold", 20),
                 text_color="black" )
        self.heading2.grid(row=8, column=0, padx=45, pady=20, sticky="w")

        self.subtitle3 = customtkinter.CTkLabel(self.scrollable_radiobutton_frame, text="Learn Alphabet: https://www.youtube.com/watch?v=iYpTJ5cEl9Y \n\nBasic FSL Phrases: https://www.youtube.com/watch?v=UytbjabYX_8 \n\nBasic Greetings: https://www.youtube.com/watch?v=iom4x_bn2MI  \n\nFSL Emergency Phrases: https://www.youtube.com/watch?v=x_zuVvOES3M  \n\nBasic Greetings: https://www.youtube.com/watch?v=iom4x_bn2MI \n\nSign Language Tutorial for Kids: https://www.youtube.com/watch?v=w13Mrs5dHZ0 \n\nFilipino Sign Language for Kids: https://www.youtube.com/watch?v=1fHAQ5UDAGo \n\nFilipino Sign Language (Kids Edition): https://www.youtube.com/watch?v=hFFx0yGf20Q", font=("Arial", 14),
        text_color="black", justify="left" )
        self.subtitle3.grid(row=9, column=0, padx=45, pady=(0, 25), sticky="nw")

        #UPDATES PAGE ----------------------------------------------------------------------------------------------------------------------------------        
        self.updates_frame = customtkinter.CTkFrame(self, corner_radius=0, fg_color="#e3e3e3", width=1400, height=700)
        self.updates_frame.grid_columnconfigure(0, weight=1)

        #yellow
        self.updates_container = customtkinter.CTkFrame(self.updates_frame, fg_color="#fff4dc", border_width=2, border_color="#FFF0D5")
        self.updates_container.grid(row=0, column=0, sticky="nsew", pady=(25, 0), padx=20, columnspan=3, rowspan=3)

        # create scrollable radiobutton frame
        self.updates_scrollable_frame = customtkinter.CTkScrollableFrame(self.updates_container, width=500,  fg_color="#fff4dc",) 
                                                                       
        self.updates_scrollable_frame.grid(row=1, column=0, padx=15, sticky="ns", pady=(25, 25))
        self.updates_scrollable_frame.configure(width=1010, height=590)
        
        self.updatesheading = customtkinter.CTkLabel(self.updates_scrollable_frame, text="SignLingu App Update - Version 1.1", font=("Arial Bold", 20),
                 text_color="black" )
        self.updatesheading.grid(row=1, column=0, padx=45, pady=(50,10), sticky="w")

        self.updatesheading1 = customtkinter.CTkLabel(self.updates_scrollable_frame, text="What's New?", font=("Arial", 20),
                 text_color="black" )
        self.updatesheading1.grid(row=2, column=0, padx=45, pady=(0,10), sticky="w")

        self.upd1 = customtkinter.CTkLabel(self.updates_scrollable_frame, text="1. Real-Time Translation for Filipino Sign Language:", font=("Arial Bold", 14),
                 text_color="black", justify="left" )
        self.upd1.grid(row=3, column=0, padx=45,  pady=(0, 5), sticky="w")

        self.upd1det = customtkinter.CTkLabel(self.updates_scrollable_frame, text="SignLingu now offers real-time translation specifically for Filipino Sign Language. Communicate seamlessly \nusing signs, and see instant text translations on the screen.", font=("Arial", 14),
                 text_color="black", justify="left" )
        self.upd1det.grid(row=4, column=0, padx=45,  pady=(0, 5), sticky="w")

        self.upd2 = customtkinter.CTkLabel(self.updates_scrollable_frame, text="2. Filipino Letters Recognition:", font=("Arial Bold", 14),
                 text_color="black", justify="left" )
        self.upd2.grid(row=5, column=0, padx=45,  pady=(0, 5), sticky="w")

        self.upd2det = customtkinter.CTkLabel(self.updates_scrollable_frame, text="Explore a new feature that allows SignLingu to recognize Filipino Sign Language letters. Improve your  \ncommunication with accurate letter recognition.", font=("Arial", 14),
                 text_color="black", justify="left" )
        self.upd2det.grid(row=6, column=0, padx=45,  pady=(0, 5), sticky="w")

        self.upd3 = customtkinter.CTkLabel(self.updates_scrollable_frame, text="3. Learn More about FSL:", font=("Arial Bold", 14),
                 text_color="black", justify="left" )
        self.upd3.grid(row=7, column=0, padx=45,  pady=(0, 5), sticky="w")

        self.upd3det = customtkinter.CTkLabel(self.updates_scrollable_frame, text="Introducing a dedicated 'Learn More about FSL' page! Enhance your signing skills with instructional videos \nand educational content. Discover the beauty and richness of Filipino Sign Language.", font=("Arial", 14),
                 text_color="black", justify="left" )
        self.upd3det.grid(row=8, column=0, padx=45,  pady=(0, 5), sticky="w")

        self.faqsheading1 = customtkinter.CTkLabel(self.updates_scrollable_frame, text="Frequently Asked Questions (FAQs)", font=("Arial Bold", 20),
                 text_color="black" )
        self.faqsheading1.grid(row=9, column=0, padx=45, pady=(20,10), sticky="w")

        self.faq1 = customtkinter.CTkLabel(self.updates_scrollable_frame, text="What is SignLingu?", font=("Arial Bold", 14),
                 text_color="black", justify="left" )
        self.faq1.grid(row=10, column=0, padx=45,  pady=(0, 5), sticky="w")

        self.faq1det = customtkinter.CTkLabel(self.updates_scrollable_frame, text="SignLingo is a desktop application designed to translate Filipino Sign Language into text in real-time. It recognizes signs,  \nincluding Filipino Sign Language letters, and provides instant text translations.", font=("Arial", 14),
                 text_color="black", justify="left" )
        self.faq1det.grid(row=11, column=0, padx=45,  pady=(0, 5), sticky="w")

        self.faq2 = customtkinter.CTkLabel(self.updates_scrollable_frame, text="How does SignLingu work?", font=("Arial Bold", 14),
                 text_color="black", justify="left" )
        self.faq2.grid(row=12, column=0, padx=45,  pady=(0, 5), sticky="w")

        self.faq2det = customtkinter.CTkLabel(self.updates_scrollable_frame, text="SignLingu uses advanced algorithms to analyze real-time sign language input. It recognizes signs in Filipino Sign   \nLanguage, translates them into text, and displays the translated content on the screen.", font=("Arial", 14),
                 text_color="black", justify="left" )
        self.faq2det.grid(row=13, column=0, padx=45,  pady=(0, 5), sticky="w")

        self.faq3 = customtkinter.CTkLabel(self.updates_scrollable_frame, text="How can I use SignLingu?", font=("Arial Bold", 14),
                 text_color="black", justify="left" )
        self.faq3.grid(row=14, column=0, padx=45,  pady=(0, 5), sticky="w")

        self.faq3det = customtkinter.CTkLabel(self.updates_scrollable_frame, text="Using SignLingu is easy! Simply launch the app, enable your webcam, and start signing in Filipino Sign Language.  The app will \nrecognize your signs in real-time and display the corresponding text translations. Explore the dedicated 'Learn More about FSL' \nand 'Filipino Sign Language' page to enrich your signing vocabulary.", font=("Arial", 14),
                 text_color="black", justify="left" )
        self.faq3det.grid(row=15, column=0, padx=45,  pady=(0, 5), sticky="w")

        self.ty = customtkinter.CTkLabel(self.updates_scrollable_frame, text="Thank you for choosing SignLingu for your Filipino Sign Language communication needs! If you have additional  \nquestions or need support, feel free to reach out to our team at c.cahilog.519535@umindanao.edu.ph, \n f.salonga.517312@umindanao.edu.ph, n.yee.517677@umindanao.edu.ph.", font=("Arial", 14),
                 text_color="black", justify="left" )
        self.ty.grid(row=16, column=0, padx=45,  pady=(15, 50), sticky="w")
        
        #ABOUTUS PAGE ----------------------------------------------------------------------------------------------------------------------------------
        aboutus_images =[
            {"path": "simulator/icons/cath.jpg", "label": "Catherine N. Cahilog\nc.cahilog.519535@umindanao.edu.ph"},
            {"path": "simulator/icons/feranz.jpg", "label": "Feranz C. Salonga\nf.salonga.517312@umindanao.edu.ph"},
            {"path": "simulator/icons/nikki.png", "label": "Nikkita A. Yee\nn.yee.517677@umindanao.edu.ph"},
        ]
        
        self.aboutus_frame = customtkinter.CTkFrame(self, corner_radius=0, fg_color="#e3e3e3", width=1400, height=700)
        self.aboutus_frame.grid_columnconfigure(0, weight=1)
        
        self.aboutus_container = customtkinter.CTkFrame(self.aboutus_frame, height=50, fg_color="#fff4dc", border_width=2, border_color="#FFF0D5")
        self.aboutus_container.grid(row=0, column=0, sticky="nsew", pady=(25, 0), padx=20, columnspan=3, rowspan=3)
        
        self.aboutus_row_container = customtkinter.CTkFrame(self.aboutus_container, fg_color="#fff4dc")
        self.aboutus_row_container.grid(row=0, column=0, columnspan=3, padx=20, pady=20)
                
        self.aboutus_mainheading = customtkinter.CTkLabel(self.aboutus_row_container, text="About SignLingu", font=("Arial Bold", 30),
                 text_color="black")
        self.aboutus_mainheading.grid(row=1, column=0, padx=45, pady=(35,0), sticky="w")
        
        self.aboutus_heading = customtkinter.CTkLabel(self.aboutus_row_container, text="Our Mission", font=("Arial Bold", 20),
                 text_color="black")
        self.aboutus_heading.grid(row=2, column=0, padx=45, pady=(20,0), sticky="w")
        
        self.aboutus_content = customtkinter.CTkLabel(self.aboutus_row_container, text="At SignLingu, we are dedicated to breaking down communication barriers for the deaf and hard-of-hearing community. Our mission is to provide a \nseamless bridge between sign language and text, empowering individuals to express themselves effortlessly and be understood by a broader \naudience.", font=("Arial", 14),
                 text_color="black", justify="left" )
        self.aboutus_content.grid(row=3, column=0, padx=45,  pady=(0,40), sticky="w")
          
        self.aboutus_column_container = customtkinter.CTkFrame(self.aboutus_row_container, fg_color="#fff4dc")
        self.aboutus_column_container.grid(row=4, column=0, padx=20, pady=20)
        
        for i, info in enumerate(aboutus_images):
            # Load the image
            original_image = Image.open(info["path"])

            # Resize and round the image
            size = (150, 150)
            rounded_image = self.round_image(original_image, size)

            # Convert to Tkinter PhotoImage
            photo_image = ImageTk.PhotoImage(rounded_image)

            # Display the image
            label_widget = tk.Label(self.aboutus_column_container, image=photo_image, bg="#fff4dc")
            label_widget.image = photo_image  # To prevent image from being garbage collected
            label_widget.grid(row=4, column=i, padx=100, pady=10)

            # Create and display label below the image
            label_text = tk.Label(self.aboutus_column_container, text=info["label"], bg="#fff4dc", font=("Arial", 14))
            label_text.grid(row=5, column=i, pady=(0, 10))
        
        self.aboutus_heading2 = customtkinter.CTkLabel(self.aboutus_row_container, text="Get In Touch", font=("Arial Bold", 20),
                 text_color="black")
        self.aboutus_heading2.grid(row=6, column=0, padx=45, pady=(40,0), sticky="w")
        
        self.aboutus_content2 = customtkinter.CTkLabel(self.aboutus_row_container, text="Whether you're a user, supporter, or potential partner, we value your input. If you have any questions, feedback, or suggestions, please reach \nout to us at info@signlingu.com. Your insights are crucial in helping us enhance the SignLingu experience.", font=("Arial", 14),
                 text_color="black", justify="left" )
        self.aboutus_content2.grid(row=7, column=0, padx=45,  pady=(0,60), sticky="w")       
        
        # select default frame
        self.select_frame_by_name("home")

    #HOME FUNCTIONS ----------------------------------------------------------------------------------------------------------------------------------
    def select_frame_by_name(self, name):
        # set button color for selected button
        self.home_button.configure(fg_color=("#f4a808", "#f4a808") if name == "home" else "#8a7d62")
        self.homeframe_button_1.configure(fg_color=("#8a7d62", "#8a7d62") if name == "rfsl_frame" else "#f4a808")
        self.dfsl_button.configure(fg_color=("#f4a808", "#f4a808") if name == "dfsl_frame" else "#8a7d62")
        self.blog_button.configure(fg_color=("#f4a808", "#f4a808") if name == "blog_frame" else "#8a7d62")
        self.update_button.configure(fg_color="transparent" if name == "updates_frame" else "transparent")
        self.aboutus_button.configure(fg_color="transparent" if name == "aboutus_frame" else "transparent")
        
        # show selected frame
        if name == "home":
            self.home_frame.grid(row=0, column=1, sticky="nsew")
        else:
            self.home_frame.grid_forget()
        if name == "dfsl_frame":
            self.dfsl_frame.grid(row=0, column=1, sticky="nsew")
        else:
            self.dfsl_frame.grid_forget()
        if name == "blog_frame":
            self.blog_frame.grid(row=0, column=1, sticky="nsew")
        else:
            self.blog_frame.grid_forget()
        if name == "rfsl_frame":
            self.rfsl_frame.grid(row=0, column=1, sticky="nsew")
        else:
            self.rfsl_frame.grid_forget()
        if name == "updates_frame":
            self.updates_frame.grid(row=0, column=1, sticky="nsew")
        else:
            self.updates_frame.grid_forget()
        if name == "aboutus_frame":
            self.aboutus_frame.grid(row=0, column=1, sticky="nsew")
        else:
            self.aboutus_frame.grid_forget()

    def home_button_event(self):
        self.select_frame_by_name("home")

    def dfsl_button_event(self):
        self.select_frame_by_name("dfsl_frame")

    def blog_button_event(self):
        self.select_frame_by_name("blog_frame")

    def rfsl_button_event(self):
        self.select_frame_by_name("rfsl_frame")

    def update_button_event(self):
        self.select_frame_by_name("updates_frame")

    def aboutus_button_event(self):
        self.select_frame_by_name("aboutus_frame")

    #DFSL DETAILS FRAME
    def show_dfsl_details_frame(self, photo_image, label_text):
        self.dfsl_frame.grid_forget()

        self.dfsl_selected_image_label.configure(image=photo_image)
        self.dfsl_label.configure(text=label_text)

        self.dfsl_details_frame.grid(row=0, column=1, sticky="nsew")
    
    def show_dfsl_frame(self):
        self.dfsl_details_frame.grid_forget()

        self.dfsl_frame.grid(row=0, column=1, sticky="nsew")
    #ROUND IMAGE
    def round_image(self, image, size):
        # Create a mask to round the image
        mask = Image.new("L", size, 0)
        draw = ImageDraw.Draw(mask)
        draw.ellipse((0, 0, size[0], size[1]), fill=255)

        # Resize the image and apply the mask
        rounded_image = image.resize(size).convert("RGBA")
        rounded_image.putalpha(mask)

        return rounded_image
    #RFSL FUNCTIONS ----------------------------------------------------------------------------------------------------------------------------------
    def toggle_play_pause(self, event):
        # Toggle the play/pause state
        self.is_playing = not self.is_playing

        # Change button color and icon accordingly
        if self.is_playing:
            self.play_pause_button.configure(fg_color="#8a7d62", hover_color="#8a7d62", border_color="#8a7d62", image=self.pauseicon)
            self.update_prediction_entry()
        else:
            self.play_pause_button.configure(fg_color="#f4a808", hover_color="#f4a808", border_color="#F4A808", image=self.playicon)
            # Clear the prediction entry
            self.prediction_entry.delete(0, "end")
                
        # Unbind the event before rebinding
        self.play_pause_button.unbind("<Button-1>")
        self.play_pause_button.bind("<Button-1>", self.toggle_play_pause)

    def update_prediction_entry(self):
        if self.predicted_class_index is not None:
            predicted_letter = self.keypoint_classifier_labels[self.predicted_class_index]
            self.prediction_entry.delete(0, "end") 
            self.prediction_entry.insert("end", predicted_letter)

    def update_camera(self):
        mode = 0
        # FPS Measurement
        fps = self.cvFpsCalc.get()
        
        # Process Key (ESC: end)
        key = cv.waitKey(10)
        if key == 27:  # ESC
            return
        number, mode = self.select_mode(key, mode)

        # Camera capture
        ret, image = self.cap.read()
    
        if not ret:
            return
                
        image = cv.flip(image, 1)  # 1 for horizontal flip
        debug_image = copy.deepcopy(cv.cvtColor(image, cv.COLOR_BGR2RGB))

        # Detection implementation
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        image[:, :, 0] = image[:, :, 0] * 0.5

        # Detection implementation
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = self.hands.process(image)
        image.flags.writeable = True

        if results.multi_hand_landmarks is not None:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks,
                                                    results.multi_handedness):
                # Bounding box calculation
                brect = self.calc_bounding_rect(debug_image, hand_landmarks)
                # Landmark calculation
                landmark_list = self.calc_landmark_list(debug_image, hand_landmarks)

                # Conversion to relative coordinates / normalized coordinates
                pre_processed_landmark_list = self.pre_process_landmark(
                    landmark_list)
                pre_processed_point_history_list = self.pre_process_point_history(
                    debug_image, self.point_history)
                # Write to the dataset file
                self.logging_csv(number, mode, pre_processed_landmark_list,
                            pre_processed_point_history_list)

                # Hand sign classification
                hand_sign_id = self.keypoint_classifier(pre_processed_landmark_list)
                if hand_sign_id == 2:  # Point gesture
                    self.point_history.append(landmark_list[8])
                else:
                    self.point_history.append([0, 0])
                
                # Hand sign classification
                hand_sign_id, hand_sign_prob = self.keypoint_classifier(pre_processed_landmark_list)
                print("CNN:", self.keypoint_classifier_labels[hand_sign_id])
                print("Probability of CNN:", hand_sign_prob)

                # Finger gesture classification
                finger_gesture_id = 0
                point_history_len = len(pre_processed_point_history_list)
                if point_history_len == (self.history_length * 2):
                    finger_gesture_id = self.point_history_classifier(
                        pre_processed_point_history_list)

                # Calculates the gesture IDs in the latest detection
                self.finger_gesture_history.append(finger_gesture_id)
                most_common_fg_id = Counter(self.finger_gesture_history).most_common()

                # Drawing part
                debug_image = self.draw_bounding_rect(self.use_brect, debug_image, brect)
                debug_image = self.draw_landmarks(debug_image, landmark_list)
                debug_image = self.draw_info_text(
                    debug_image,
                    brect,
                    handedness,
                    self.keypoint_classifier_labels[hand_sign_id],
                    self.point_history_classifier_labels[most_common_fg_id[0][0]],
                )

                # Find nearest classes
                if hand_sign_id != -1:
                    data_point = np.array(pre_processed_landmark_list)
                    nearest_classes = self.find_nearest_classes(data_point, self.X, self.y)
                    print("KNN:", nearest_classes)

                    nearest_data_points = []
                    for cls in nearest_classes:
                        # Convert class labels back to numeric format if needed
                        cls_label = ord(cls) - ord('A')  # Assuming classes are A, B, C, ...
                        nearest_data_points.append(self.X[self.y == cls_label][0]) 

                    # Use the trained Random Forest classifier to predict the class
                    predicted_class = self.rf_model.predict(nearest_data_points)
                    self.predicted_class_index = predicted_class[0]  # Index of the predicted class
                    probabilities = self.rf_model.predict_proba(nearest_data_points)
                    predicted_class_probability = probabilities[0][self.predicted_class_index]  # Probability of the predicted class
                    print("Probabilities of the 5 nearest classes:")
                    for i, cls in enumerate(nearest_classes):
                        print(f"{cls}: {probabilities[0][i]}")
                    print("Random Forest predicted class:", self.keypoint_classifier_labels[self.predicted_class_index])
                    print("Probability of the Random Forest predicted class:", predicted_class_probability)
            
                    #time.sleep(1) #testing

                if self.is_playing:
                    self.update_prediction_entry()

        else:
            self.point_history.append([0, 0])

        debug_image = self.draw_point_history(debug_image, self.point_history)

        # Display the processed image
        img = Image.fromarray(debug_image)
        img = ImageTk.PhotoImage(img)
        self.camera_label.configure(image=img)  
        self.camera_label.image = img

        self.camera_label.after(10, self.update_camera)

    def start_camera(self):
        self.cap = cv.VideoCapture(0)

        if self.cap.isOpened():
            self.update_camera()

    def stop_camera(self):
        self.camera_label.configure(image="")
        if hasattr(self, 'cap') and self.cap.isOpened():
            self.cap.release()

    def mainloop(self):
        super().mainloop()
        self.stop_camera()

    def load_dataset(self, file_path):
        with open(file_path, 'r') as f:
            reader = csv.reader(f)
            data = list(reader)
        X = np.array([row[1:] for row in data], dtype=float)
        y = np.array([int(row[0]) for row in data])
        return X, y
    
    def find_nearest_classes(self, data_point, X, y):
        distances = []
        for i, x in enumerate(X):
            distance = np.linalg.norm(data_point - x)  # Euclidean distance
            distances.append((distance, y[i]))
        distances.sort(key=lambda x: x[0])  # Sort by distance
        nearest_classes = [chr(ord('A') + cls) for _, cls in distances[:5]]
        return nearest_classes
    
    def get_args(self):
        parser = argparse.ArgumentParser()

        parser.add_argument("--device", type=int, default=0)
        parser.add_argument("--width", help='cap width', type=int, default=960)
        parser.add_argument("--height", help='cap height', type=int, default=540)

        parser.add_argument('--use_static_image_mode', action='store_true')
        parser.add_argument("--min_detection_confidence",
                            help='min_detection_confidence',
                            type=float,
                            default=0.7)
        parser.add_argument("--min_tracking_confidence",
                            help='min_tracking_confidence',
                            type=int,
                            default=0.5)

        args = parser.parse_args()

        return args
    
    def select_mode(self, key, mode):
        number = -1
        if 48 <= key <= 57:  # 0 ~ 9
            number = key - 48
        if key == 110:  # n
            mode = 0
        if key == 107:  # k
            mode = 1
        if key == 104:  # h
            mode = 2
        return number, mode

    def calc_bounding_rect(self, image, landmarks):
        image_width, image_height = image.shape[1], image.shape[0]

        landmark_array = np.empty((0, 2), int)

        for _, landmark in enumerate(landmarks.landmark):
            landmark_x = min(int(landmark.x * image_width), image_width - 1)
            landmark_y = min(int(landmark.y * image_height), image_height - 1)

            landmark_point = [np.array((landmark_x, landmark_y))]

            landmark_array = np.append(landmark_array, landmark_point, axis=0)

        x, y, w, h = cv.boundingRect(landmark_array)

        return [x, y, x + w, y + h]
    
    def calc_landmark_list(self, image, landmarks):
        image_width, image_height = image.shape[1], image.shape[0]

        landmark_point = []

        # Keypoint
        for _, landmark in enumerate(landmarks.landmark):
            landmark_x = min(int(landmark.x * image_width), image_width - 1)
            landmark_y = min(int(landmark.y * image_height), image_height - 1)
            # landmark_z = landmark.z

            landmark_point.append([landmark_x, landmark_y])

        return landmark_point
    
    def pre_process_landmark(self, landmark_list):
        temp_landmark_list = copy.deepcopy(landmark_list)

        # Convert to relative coordinates
        base_x, base_y = 0, 0
        for index, landmark_point in enumerate(temp_landmark_list):
            if index == 0:
                base_x, base_y = landmark_point[0], landmark_point[1]

            temp_landmark_list[index][0] = temp_landmark_list[index][0] - base_x
            temp_landmark_list[index][1] = temp_landmark_list[index][1] - base_y

        # Convert to a one-dimensional list
        temp_landmark_list = list(
            itertools.chain.from_iterable(temp_landmark_list))

        # Normalization
        max_value = max(list(map(abs, temp_landmark_list)))

        def normalize_(n):
            return n / max_value

        temp_landmark_list = list(map(normalize_, temp_landmark_list))

        return temp_landmark_list
    
    def pre_process_point_history(self, image, point_history):
        image_width, image_height = image.shape[1], image.shape[0]

        temp_point_history = copy.deepcopy(point_history)

        # Convert to relative coordinates
        base_x, base_y = 0, 0
        for index, point in enumerate(temp_point_history):
            if index == 0:
                base_x, base_y = point[0], point[1]

            temp_point_history[index][0] = (temp_point_history[index][0] -
                                            base_x) / image_width
            temp_point_history[index][1] = (temp_point_history[index][1] -
                                            base_y) / image_height

        # Convert to a one-dimensional list
        temp_point_history = list(
            itertools.chain.from_iterable(temp_point_history))

        return temp_point_history

    def logging_csv(self, number, mode, landmark_list, point_history_list):
        if mode == 0:
            pass
        if mode == 1 and (0 <= number <= 9):
            csv_path = 'model/keypoint_classifier/keypoint.csv'
            with open(csv_path, 'a', newline="") as f:
                writer = csv.writer(f)
                writer.writerow([number, *landmark_list])
        if mode == 2 and (0 <= number <= 9):
            csv_path = 'model/point_history_classifier/point_history.csv'
            with open(csv_path, 'a', newline="") as f:
                writer = csv.writer(f)
                writer.writerow([number, *point_history_list])
        return

    def draw_transparent_line(self, image, start_point, end_point, color, alpha, thickness):
        overlay = image.copy()
        cv.line(overlay, start_point, end_point, color, thickness, cv.LINE_AA)
        cv.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)
    
    def draw_transparent_circle(self, image, center, radius, color, alpha, thickness):
        overlay = image.copy()
        cv.circle(overlay, center, radius, color, thickness, cv.LINE_AA)
        cv.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)

    def draw_landmarks(self, image, landmark_point, alpha=0, thickness=2):
        if len(landmark_point) > 0:
            # Thumb
            self.draw_transparent_line(image, tuple(landmark_point[2]), tuple(landmark_point[3]),
                                (0, 0, 0), alpha, 6)
            self.draw_transparent_line(image, tuple(landmark_point[2]), tuple(landmark_point[3]),
                                (255, 255, 255), alpha, 2)
            self.draw_transparent_line(image, tuple(landmark_point[3]), tuple(landmark_point[4]),
                                (0, 0, 0), alpha, 6)
            self.draw_transparent_line(image, tuple(landmark_point[3]), tuple(landmark_point[4]),
                                (255, 255, 255), alpha, 2)
            
            # Index finger
            self.draw_transparent_line(image, tuple(landmark_point[5]), tuple(landmark_point[6]),
                                (0, 0, 0), alpha, 6)
            self.draw_transparent_line(image, tuple(landmark_point[5]), tuple(landmark_point[6]),
                                (255, 255, 255), alpha, 2)
            self.draw_transparent_line(image, tuple(landmark_point[6]), tuple(landmark_point[7]),
                                (0, 0, 0), alpha, 6)
            self.draw_transparent_line(image, tuple(landmark_point[6]), tuple(landmark_point[7]),
                                (255, 255, 255), alpha, 2)
            self.draw_transparent_line(image, tuple(landmark_point[7]), tuple(landmark_point[8]),
                                (0, 0, 0), alpha, 6)
            self.draw_transparent_line(image, tuple(landmark_point[7]), tuple(landmark_point[8]),
                                (255, 255, 255), alpha, 2)

            # Middle finger
            self.draw_transparent_line(image, tuple(landmark_point[9]), tuple(landmark_point[10]),
                                (0, 0, 0), alpha, 6)
            self.draw_transparent_line(image, tuple(landmark_point[9]), tuple(landmark_point[10]),
                                (255, 255, 255), alpha, 2)
            self.draw_transparent_line(image, tuple(landmark_point[10]), tuple(landmark_point[11]),
                                (0, 0, 0), alpha, 6)
            self.draw_transparent_line(image, tuple(landmark_point[10]), tuple(landmark_point[11]),
                                (255, 255, 255), alpha, 2)
            self.draw_transparent_line(image, tuple(landmark_point[11]), tuple(landmark_point[12]),
                                (0, 0, 0), alpha, 6)
            self.draw_transparent_line(image, tuple(landmark_point[11]), tuple(landmark_point[12]),
                                (255, 255, 255), alpha, 2)

            # Ring finger
            self.draw_transparent_line(image, tuple(landmark_point[13]), tuple(landmark_point[14]),
                                (0, 0, 0), alpha, 6)
            self.draw_transparent_line(image, tuple(landmark_point[13]), tuple(landmark_point[14]),
                                (255, 255, 255), alpha, 2)
            self.draw_transparent_line(image, tuple(landmark_point[14]), tuple(landmark_point[15]),
                                (0, 0, 0), alpha, 6)
            self.draw_transparent_line(image, tuple(landmark_point[14]), tuple(landmark_point[15]),
                                (255, 255, 255), alpha, 2)
            self.draw_transparent_line(image, tuple(landmark_point[15]), tuple(landmark_point[16]),
                                (0, 0, 0), alpha, 6)
            self.draw_transparent_line(image, tuple(landmark_point[15]), tuple(landmark_point[16]),
                                (255, 255, 255), alpha, 2)

            # Little finger
            self.draw_transparent_line(image, tuple(landmark_point[17]), tuple(landmark_point[18]),
                                (0, 0, 0), alpha, 6)
            self.draw_transparent_line(image, tuple(landmark_point[17]), tuple(landmark_point[18]),
                                (255, 255, 255), alpha, 2)
            self.draw_transparent_line(image, tuple(landmark_point[18]), tuple(landmark_point[19]),
                                (0, 0, 0), alpha, 6)
            self.draw_transparent_line(image, tuple(landmark_point[18]), tuple(landmark_point[19]),
                                (255, 255, 255), alpha, 2)
            self.draw_transparent_line(image, tuple(landmark_point[19]), tuple(landmark_point[20]),
                                (0, 0, 0), alpha, 6)
            self.draw_transparent_line(image, tuple(landmark_point[19]), tuple(landmark_point[20]),
                                (255, 255, 255), alpha, 2)

            # Palm
            self.draw_transparent_line(image, tuple(landmark_point[0]), tuple(landmark_point[1]),
                                (0, 0, 0), alpha, 6)
            self.draw_transparent_line(image, tuple(landmark_point[0]), tuple(landmark_point[1]),
                                (255, 255, 255), alpha, 2)
            self.draw_transparent_line(image, tuple(landmark_point[1]), tuple(landmark_point[2]),
                                (0, 0, 0), alpha, 6)
            self.draw_transparent_line(image, tuple(landmark_point[1]), tuple(landmark_point[2]),
                                (255, 255, 255), alpha, 2)
            self.draw_transparent_line(image, tuple(landmark_point[2]), tuple(landmark_point[5]),
                                (0, 0, 0), alpha, 6)
            self.draw_transparent_line(image, tuple(landmark_point[2]), tuple(landmark_point[5]),
                                (255, 255, 255), alpha, 2)
            self.draw_transparent_line(image, tuple(landmark_point[5]), tuple(landmark_point[9]),
                                (0, 0, 0), alpha, 6)
            self.draw_transparent_line(image, tuple(landmark_point[5]), tuple(landmark_point[9]),
                                (255, 255, 255), alpha, 2)
            self.draw_transparent_line(image, tuple(landmark_point[9]), tuple(landmark_point[13]),
                                (0, 0, 0), alpha, 6)
            self.draw_transparent_line(image, tuple(landmark_point[9]), tuple(landmark_point[13]),
                                (255, 255, 255), alpha, 2)
            self.draw_transparent_line(image, tuple(landmark_point[13]), tuple(landmark_point[17]),
                                (0, 0, 0), alpha, 6)
            self.draw_transparent_line(image, tuple(landmark_point[13]), tuple(landmark_point[17]),
                                (255, 255, 255), alpha, 2)
            self.draw_transparent_line(image, tuple(landmark_point[17]), tuple(landmark_point[0]),
                                (0, 0, 0), alpha, 6)
            self.draw_transparent_line(image, tuple(landmark_point[17]), tuple(landmark_point[0]),
                                (255, 255, 255), alpha, 2)

        # Key Points
        for index, landmark in enumerate(landmark_point):
            if index == 0: 
                self.draw_transparent_circle(image, tuple(landmark), 5, (255, 255, 255), alpha, thickness)
                self.draw_transparent_circle(image, tuple(landmark), 5, (0, 0, 0), alpha, thickness)
            if index == 1:  
                self.draw_transparent_circle(image, tuple(landmark), 5, (255, 255, 255), alpha, thickness)
                self.draw_transparent_circle(image, tuple(landmark), 5, (0, 0, 0), alpha, thickness)
            if index == 2:  
                self.draw_transparent_circle(image, tuple(landmark), 5, (255, 255, 255), alpha, thickness)
                self.draw_transparent_circle(image, tuple(landmark), 5, (0, 0, 0), alpha, thickness)
            if index == 3: 
                self.draw_transparent_circle(image, tuple(landmark), 5, (255, 255, 255), alpha, thickness)
                self.draw_transparent_circle(image, tuple(landmark), 5, (0, 0, 0), alpha, thickness)
            if index == 4:  
                self.draw_transparent_circle(image, tuple(landmark), 5, (255, 255, 255), alpha, thickness)
                self.draw_transparent_circle(image, tuple(landmark), 5, (0, 0, 0), alpha, thickness)
            if index == 5:  
                self.draw_transparent_circle(image, tuple(landmark), 5, (255, 255, 255), alpha, thickness)
                self.draw_transparent_circle(image, tuple(landmark), 5, (0, 0, 0), alpha, thickness)
            if index == 6: 
                self.draw_transparent_circle(image, tuple(landmark), 5, (255, 255, 255), alpha, thickness)
                self.draw_transparent_circle(image, tuple(landmark), 5, (0, 0, 0), alpha, thickness)
            if index == 7:  
                self.draw_transparent_circle(image, tuple(landmark), 5, (255, 255, 255), alpha, thickness)
                self.draw_transparent_circle(image, tuple(landmark), 5, (0, 0, 0), alpha, thickness)
            if index == 8:  
                self.draw_transparent_circle(image, tuple(landmark), 5, (255, 255, 255), alpha, thickness)
                self.draw_transparent_circle(image, tuple(landmark), 5, (0, 0, 0), alpha, thickness)
            if index == 9: 
                self.draw_transparent_circle(image, tuple(landmark), 5, (255, 255, 255), alpha, thickness)
                self.draw_transparent_circle(image, tuple(landmark), 5, (0, 0, 0), alpha, thickness)
            if index == 10:  
                self.draw_transparent_circle(image, tuple(landmark), 5, (255, 255, 255), alpha, thickness)
                self.draw_transparent_circle(image, tuple(landmark), 5, (0, 0, 0), alpha, thickness)
            if index == 11:  
                self.draw_transparent_circle(image, tuple(landmark), 5, (255, 255, 255), alpha, thickness)
                self.draw_transparent_circle(image, tuple(landmark), 5, (0, 0, 0), alpha, thickness)
            if index == 12:  
                self.draw_transparent_circle(image, tuple(landmark), 5, (255, 255, 255), alpha, thickness)
                self.draw_transparent_circle(image, tuple(landmark), 5, (0, 0, 0), alpha, thickness)
            if index == 13:  
                self.draw_transparent_circle(image, tuple(landmark), 5, (255, 255, 255), alpha, thickness)
                self.draw_transparent_circle(image, tuple(landmark), 5, (0, 0, 0), alpha, thickness)
            if index == 14: 
                self.draw_transparent_circle(image, tuple(landmark), 5, (255, 255, 255), alpha, thickness)
                self.draw_transparent_circle(image, tuple(landmark), 5, (0, 0, 0), alpha, thickness)
            if index == 15:  
                self.draw_transparent_circle(image, tuple(landmark), 5, (255, 255, 255), alpha, thickness)
                self.draw_transparent_circle(image, tuple(landmark), 5, (0, 0, 0), alpha, thickness)
            if index == 16: 
                self.draw_transparent_circle(image, tuple(landmark), 5, (255, 255, 255), alpha, thickness)
                self.draw_transparent_circle(image, tuple(landmark), 5, (0, 0, 0), alpha, thickness)
            if index == 17:  
                self.draw_transparent_circle(image, tuple(landmark), 5, (255, 255, 255), alpha, thickness)
                self.draw_transparent_circle(image, tuple(landmark), 5, (0, 0, 0), alpha, thickness)
            if index == 18: 
                self.draw_transparent_circle(image, tuple(landmark), 5, (255, 255, 255), alpha, thickness)
                self.draw_transparent_circle(image, tuple(landmark), 5, (0, 0, 0), alpha, thickness)
            if index == 19:  
                self.draw_transparent_circle(image, tuple(landmark), 5, (255, 255, 255), alpha, thickness)
                self.draw_transparent_circle(image, tuple(landmark), 5, (0, 0, 0), alpha, thickness)
            if index == 20:  
                self.draw_transparent_circle(image, tuple(landmark), 5, (255, 255, 255), alpha, thickness)
                self.draw_transparent_circle(image, tuple(landmark), 5, (0, 0, 0), alpha, thickness)
            
        return image    
    
    def draw_bounding_rect(self, use_brect, image, brect):
        if use_brect:
            # Outer rectangle
            cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[3]),
                        (225, 225, 225), 1)

        return image

    def draw_info_text(self, image, brect, handedness, hand_sign_text,
                    finger_gesture_text):
        info_text = handedness.classification[0].label[0:]
        if finger_gesture_text != "":
            cv.putText(image, "" + finger_gesture_text + ': ' + info_text, (40, 50),
                    cv.FONT_HERSHEY_DUPLEX, 0.7, (0, 0, 0), 1, cv.LINE_AA)
            cv.putText(image, "" + finger_gesture_text + ': ' + info_text, (40, 50),
                    cv.FONT_HERSHEY_DUPLEX, 0.7, (255, 255, 255), 1, cv.LINE_AA)


        return image

    def draw_point_history(self, image, point_history):
        for index, point in enumerate(point_history):
            if point[0] != 0 and point[1] != 0:
                cv.circle(image, (point[0], point[1]), 1 + int(index / 2),
                        (152, 251, 152), 2)

        return image   

if __name__ == "__main__":
    app = App()
    app.mainloop()

