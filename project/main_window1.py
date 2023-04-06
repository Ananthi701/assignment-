import tkinter as tk
import numpy as np
from tkinter import *
import tkinter.font as tkFont
import os
from tkinter import filedialog as fd 
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
from tensorflow.keras.optimizers import RMSprop
from tensorflow import keras
import numpy as np
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from tkinter import messagebox as messagebox
class App:
    
    def __init__(self):
        #setting title
        self.root = tk.Tk()
        self.root.title("undefined")
        #setting window size
        width=1053
        height=648
        screenwidth = self.root.winfo_screenwidth()
        screenheight = self.root.winfo_screenheight()
        alignstr = '%dx%d+%d+%d' % (width, height, (screenwidth - width) / 2, (screenheight - height) / 2)
        self.root.geometry(alignstr)
        self.root.resizable(width=False, height=False)

        GButton_626=tk.Button(self.root)
        GButton_626["bg"] = "#1e90ff"
        ft = tkFont.Font(family='Times',size=10)
        GButton_626["font"] = ft
        GButton_626["fg"] = "#ffffff"
        GButton_626["justify"] = "center"
        GButton_626["text"] = "Load Image"
        GButton_626.place(x=30,y=140,width=181,height=47)
        GButton_626["command"] = self.GButton_626_command

        self.GButton_633=tk.Button(self.root)
        self.GButton_633["bg"] = "#FC9007"
        ft = tkFont.Font(family='Times',size=10)
        self.GButton_633["font"] = ft
        self.GButton_633["fg"] = "#FFFFFF"
        self.GButton_633["justify"] = "center"
        self.GButton_633["text"] = "Result"
        self.GButton_633.place(x=30,y=200,width=180,height=46)
        self.GButton_633["command"] = self.GButton_633_command
        self.GButton_633["state"] = "disabled"

        

        GButton_432=tk.Button(self.root)
        GButton_432["bg"] = "#FF5780"
        ft = tkFont.Font(family='Times',size=10)
        GButton_432["font"] = ft
        GButton_432["fg"] = "#ffffff"
        GButton_432["justify"] = "center"
        GButton_432["text"] = "Exit"
        GButton_432.place(x=30,y=500,width=180,height=44)
        GButton_432["command"] = self.GButton_432_command

        GLabel_480=tk.Label(self.root)
        ft = tkFont.Font(family='Times',size=22)
        GLabel_480["font"] = ft
        GLabel_480["fg"] = "#333333"
        GLabel_480["justify"] = "center"
        GLabel_480["text"] = "Lung Cancer Detection Using Deep Learning"
        GLabel_480.place(x=0,y=30,width=1052,height=30)

        GLabel_232=tk.Label(self.root)
        ft = tkFont.Font(family='Times',size=10)
        GLabel_232["font"] = ft
        GLabel_232["fg"] = "#333333"
        GLabel_232["justify"] = "center"
        GLabel_232["text"] = ""
        GLabel_232.place(x=240,y=140,width=716,height=306)

        self.GLabel_552=tk.Label(self.root)
        ft = tkFont.Font(family='Times',size=16)
        self.GLabel_552["font"] = ft
        self.GLabel_552["fg"] = "#333333"
        self.GLabel_552["justify"] = "center"
        self.GLabel_552["text"] = ""
        self.GLabel_552.place(x=290,y=510,width=170,height=25)

        self.root.mainloop()

    def GButton_626_command(self):
        print("command")
        self.GButton_633["state"] = "normal"
        self.path= fd.askopenfilename( initialdir=os.getcwd(),title = "Select a File", 
                                          filetypes = (("Image","*.png"),("all files","*.*"))) 
        self.file_size = os.path.getsize(self.path)
        print(self.path)
        print(self.file_size)
        self.fname=os.path.basename(self.path)
        print(self.fname)
        img = Image.open(self.path)
        resized = img.resize((300, 300))

# Convert the image to a PhotoImage object
        photo = ImageTk.PhotoImage(resized)

# Create a label to display the image
        label = Label(self.root, image=photo)
        label.image = photo
        label.place(x=300, y=200)


    def GButton_633_command(self):
        print("command")
        '''self.create_model()
        messagebox.showinfo("Cancer","Model Created Successfully")
        self.GButton_289["state"] = "normal"'''
        img = cv2.imread(self.path, cv2.IMREAD_GRAYSCALE)

        # Perform histogram equalization
        equalized_img = cv2.equalizeHist(img)

        # Save or display the equalized image
        cv2.imwrite('histogram.jpg', equalized_img)
      
        img = Image.open('histogram.jpg')
        resized = img.resize((300, 300))
        photo = ImageTk.PhotoImage(resized)
        label = Label(self.root, image=photo)
        label.image = photo
        label.place(x=300, y=200)
        
        img = cv2.imread('histogram.jpg', cv2.IMREAD_GRAYSCALE)

        # Perform edge detection using the Canny edge detector
        edges = cv2.Canny(img, 100, 200)

        # Save or display the edge map
        cv2.imwrite('edges.jpg', edges)
        img = Image.open('edges.jpg')
        resized = img.resize((300, 300))
        photo = ImageTk.PhotoImage(resized)
        label = Label(self.root, image=photo)
        label.image = photo
        label.place(x=300, y=200)
        img=image.load_img("histogram.jpg",target_size=(200,200))
        plt.imshow(img)
        plt.title("Grey Scale Process")
        plt.show()

        img=image.load_img('edges.jpg',target_size=(200,200))
        plt.imshow(img)
        plt.title("Edge Detection")
        plt.show()
        
        #Segmentation
        print("command")
        img = cv2.imread(self.path)
        b,g,r = cv2.split(img)
        rgb_img = cv2.merge([r,g,b])
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
        # noise removal
        kernel = np.ones((2,2),np.uint8)
        #opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 2)
        closing = cv2.morphologyEx(thresh,cv2.MORPH_CLOSE,kernel, iterations = 2)
        # sure background area
        sure_bg = cv2.dilate(closing,kernel,iterations=3)
        # Finding sure foreground area
        dist_transform = cv2.distanceTransform(sure_bg,cv2.DIST_L2,3)
        # Threshold
        ret, sure_fg = cv2.threshold(dist_transform,0.1*dist_transform.max(),255,0)
        # Finding unknown region
        sure_fg = np.uint8(sure_fg)
        unknown = cv2.subtract(sure_bg,sure_fg)
        # Marker labelling
        ret, markers = cv2.connectedComponents(sure_fg)
        # Add one to all labels so that sure background is not 0, but 1
        markers = markers+1
        # Now, mark the region of unknown with zero
        markers[unknown==255] = 0
        markers = cv2.watershed(img,markers)
        img[markers == -1] = [255,0,0]
        plt.subplot(211),plt.imshow(rgb_img)
        plt.title('Input Image'), plt.xticks([]), plt.yticks([])
        plt.subplot(212),plt.imshow(thresh, 'gray')
        plt.imsave(r'thresh.png',thresh)
        plt.title("Segment Process 1"), plt.xticks([]), plt.yticks([])
        plt.tight_layout()
        plt.show()
        img = cv2.imread(r'thresh.png')
        b,g,r = cv2.split(img)
        rgb_img = cv2.merge([r,g,b])
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
        plt.subplot(211),plt.imshow(closing, 'gray')
        plt.title("Segment 2"), plt.xticks([]), plt.yticks([])
        plt.subplot(212),plt.imshow(sure_bg, 'gray')
        plt.imsave(r'dilation.png',sure_bg)
        plt.title("Segment 3"), plt.xticks([]), plt.yticks([])
        plt.tight_layout()
        plt.show()

        


   


    def GButton_825_command(self):
        print("command")
        img = cv2.imread('histogram.jpg', cv2.IMREAD_GRAYSCALE)

        # Perform edge detection using the Canny edge detector
        edges = cv2.Canny(img, 100, 200)

        # Save or display the edge map
        cv2.imwrite('edges.jpg', edges)
        img = Image.open('edges.jpg')
        resized = img.resize((300, 300))
        photo = ImageTk.PhotoImage(resized)
        label = Label(self.root, image=photo)
        label.image = photo
        label.place(x=300, y=200)
        #self.GButton_769["state"] = "normal"


    

    def GButton_464_command(self):
        print("command")
        res=self.predict_cancer(self.path)
        
        if res=="Cancer":
            self.GLabel_552["fg"] = "#F40D0D"
            self.GLabel_552["text"] = "Cancer Found"
        else:
            self.GLabel_552["fg"] = "#029D1E"
            self.GLabel_552["text"] = res



    def GButton_432_command(self):
        print("command")
        answer = messagebox.askyesno("Exit","Do you want to Exit ?")
        print(answer)
        if answer==True:
            self.root.destroy()
    def create_model(self):
        # Define the CNN model
        '''model = keras.models.Sequential([
            keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(200, 200, 3)),
            keras.layers.MaxPooling2D(2,2),
            keras.layers.Conv2D(64, (3,3), activation='relu'),
            keras.layers.MaxPooling2D(2,2),
            keras.layers.Conv2D(128, (3,3), activation='relu'),
            keras.layers.MaxPooling2D(2,2),
            keras.layers.Conv2D(128, (3,3), activation='relu'),
            keras.layers.MaxPooling2D(2,2),
            keras.layers.Flatten(),
            keras.layers.Dense(512, activation='relu'),
            keras.layers.Dense(1, activation='sigmoid')
        ])

        # Compile the model
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

        # Prepare the data
        train_datagen = ImageDataGenerator(rescale=1./255)
        val_datagen = ImageDataGenerator(rescale=1./255)

        train_generator = train_datagen.flow_from_directory(
                'dataset/train',
                target_size=(200, 200),
                batch_size=20,
                class_mode='binary')

        val_generator = val_datagen.flow_from_directory(
                'dataset/validation',
                target_size=(200, 200),
                batch_size=20,
                class_mode='binary')
        print(train_generator.class_indices)
        print(train_generator.classes)
        # Train the model
        history = model.fit(
            train_generator,
            steps_per_epoch=100,
            epochs=100,
            validation_data=val_generator,
            validation_steps=50,
            verbose=2)'''
        train=ImageDataGenerator(rescale=1/255)
        validation=ImageDataGenerator(rescale=1/255)
        train_dataset=train.flow_from_directory('dataset/train/',
                                       target_size=(200,200),
                                       batch_size=3,
                                       class_mode='binary')
        validation_dataset=train.flow_from_directory('dataset/validation/',
                                       target_size=(200,200),
                                       batch_size=3,
                                       class_mode='binary')
        print(train_dataset.class_indices)
        print(train_dataset.classes)
        model=tf.keras.models.Sequential([tf.keras.layers.Conv2D(16,(3,3),activation='relu',input_shape=(200,200,3)),
                                 tf.keras.layers.MaxPool2D(2,2),
                                 tf.keras.layers.Conv2D(32,(3,3),activation='relu'),
                                 tf.keras.layers.MaxPool2D(2,2),
                                 tf.keras.layers.Conv2D(64,(3,3),activation='relu'),
                                 tf.keras.layers.MaxPool2D(2,2),
                                 tf.keras.layers.Flatten(),
                                 tf.keras.layers.Dense(512,activation='relu'),
                                 tf.keras.layers.Dense(1,activation='sigmoid')
                                  
                                 ])
        model.compile(loss='binary_crossentropy',
             optimizer=RMSprop(lr=0.001),
             metrics=['accuracy'])
        model_fit=model.fit(train_dataset,
                   steps_per_epoch=5,
                   epochs=30,
                   validation_data=validation_dataset)
        # Save the model
        model.save('cancer_normal.h5')
    def extract_features(self,image_path):
        # Load the image
        img = image.load_img(image_path, target_size=(224, 224))
        
        # Convert the image to a numpy array
        x = image.img_to_array(img)
        
        # Reshape the array to (1,224,224,3)
        x = np.expand_dims(x, axis=0)
        
        # Preprocess the input for VGG16
        x = preprocess_input(x)
        
        # Load the VGG16 model
        model = VGG16(weights='imagenet', include_top=False)
        
        # Extract features from the input image
        features = model.predict(x)
        
        # Flatten the features array
        features = features.flatten()
        
        return features
    def predict_cancer(self,img_path):
        '''train=ImageDataGenerator(rescale=1/255)
        validation=ImageDataGenerator(rescale=1/255)
        train_dataset=train.flow_from_directory('dataset/train/',
                                       target_size=(200,200),
                                       batch_size=3,
                                       class_mode='binary')
        validation_dataset=train.flow_from_directory('dataset/validation/',
                                       target_size=(200,200),
                                       batch_size=3,
                                       class_mode='binary')
        print(train_dataset.class_indices)
        print(train_dataset.classes)
        model=tf.keras.models.Sequential([tf.keras.layers.Conv2D(16,(3,3),activation='relu',input_shape=(200,200,3)),
                                 tf.keras.layers.MaxPool2D(2,2),
                                 tf.keras.layers.Conv2D(32,(3,3),activation='relu'),
                                 tf.keras.layers.MaxPool2D(2,2),
                                 tf.keras.layers.Conv2D(64,(3,3),activation='relu'),
                                 tf.keras.layers.MaxPool2D(2,2),
                                 tf.keras.layers.Flatten(),
                                 tf.keras.layers.Dense(512,activation='relu'),
                                 tf.keras.layers.Dense(1,activation='sigmoid')
                                  
                                 ])
        model.compile(loss='binary_crossentropy',
             optimizer=RMSprop(lr=0.001),
             metrics=['accuracy'])
        model_fit=model.fit(train_dataset,
                   steps_per_epoch=5,
                   epochs=30,
                   validation_data=validation_dataset)'''
        
        model = keras.models.load_model('cancer_normal.h5')
        img=image.load_img(self.path,target_size=(200,200))
       
        X=image.img_to_array(img)
        X=np.expand_dims(X,axis=0)
        images=np.vstack([X])
        val=model.predict(images)
        res=""
        if val == 0:
            print("Cancer")
            res= "Cancer"
        
        else:
            print("Normal")
            res ="Normal"
        
        plt.imshow(img)
        plt.show()
        return res
if __name__ == "__main__":
    
    app = App()
    
