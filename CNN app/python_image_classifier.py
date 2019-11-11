#GUI imports==========================================================================================================
# Importing Kivy - GUI library
from kivy.app import App # Imports App Class
from kivy.lang import Builder # Imports Builder to load .kv file
from kivy.uix.floatlayout import FloatLayout # Imports FloatLayout for Graphics
from kivy.graphics import Color, Rectangle # Imports graphics objects
from kivy.uix.screenmanager import ScreenManager, Screen # Imports UI manager
from kivy.uix.image import Image
from kivy.uix.label import Label # Imports Label
from kivy.uix.popup import Popup
from kivy.uix.button import Button

# Import Tkinter - GUI library
from tkinter import Tk
from tkinter.filedialog import askopenfilename
from kivy.core.window import Window
from kivy.uix.widget import Widget

# Set window size
Window.size = (900, 600)

#CNN model code========================================================================================================
# Disables AVX/FMA and TensorFlow AVX warning message
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Imports imaging and mathematical libraries for image analysis
import matplotlib.pyplot as plt
from skimage.transform import resize
import numpy as np

# Load the data
from keras.datasets import cifar10
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# One-Hot Encoding: Convert the labels into a set of 10 numbers to input into the neural network
from keras.utils import to_categorical
y_train_one_hot = to_categorical(y_train)
y_test_one_hot = to_categorical(y_test)

# Load model
from keras.models import load_model
model = load_model('imgcls_model5.h5')

# GUI===================================================================================================================
# Root Window
class RootScreen(ScreenManager):
    pass

# Start window class
class StartWindow(Screen):

    # Navigation bar links
    def next_page(self):
        print("Moving to the main page")
        screen_manager.current = "info"

class InfoWindow(Screen):
    pass

# Home Window
class CNNWindow(Screen):

    # Global variables that can be accessed by the different methods
    global outcomes_text
    my_image = None
    index = None
    class_indexes = None
    probabilities = None
    outcomes_text = None
    results = None

    # Imports and analyses the image
    def import_image(self):
        try:
            # Makes the variables global
            global my_image

            # Opens a window where you can import an image file
            Tk().withdraw()  # we don't want a full GUI, so keep the root window from appearing
            filename = askopenfilename()  # show an "Open" dialog box and return the path to the selected file

            # Show image on screen
            with self.canvas:
                Rectangle(source=filename, pos=(70, 120), size=(430, 430))

            # Load the image file
            my_image = plt.imread(filename)
            print("Image has been imported")

            return my_image

        except:
            error = "Please choose a suitable image"
            open_message(error)

    def classify_image(self):

        # Makes the variables global
        global index
        global class_indexes
        global probabilities
        global outcomes_text
        global results

        # Clears the outcomes results after every analysis
        if outcomes_text != None:
            self.remove_widget(outcomes_text)

        try:
            # Resizing the image
            my_image_resized = resize(my_image, (32, 32, 3))
            img = plt.imshow(my_image_resized)
            print("Image has been resized")

            print("Analysing image...")

            # Get probabilities for each class
            probabilities = model.predict(np.array([my_image_resized]))

            # Class array
            class_indexes = [x.upper() for x in ['airplane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse',
                                                 'ship', 'truck']]

            # Sort in Ascending order
            index = np.argsort(probabilities[0, :])

            # Outcomes
            results = f"""
            ==================================
            Most likely class: {class_indexes[index[9]]}
            | Probability: {'%.3f'%(probabilities[0, index[9]])}%
        
            Second most likely class: {class_indexes[index[8]]} 
            | Probability: {'%.3f'%(probabilities[0, index[8]])}%
        
            Third most likely class: {class_indexes[index[7]]} 
            | Probability: {'%.3f'%(probabilities[0, index[7]])}%
        
            Fourth most likely class: {class_indexes[index[6]]} 
            | Probability: {'%.3f'%(probabilities[0, index[6]])}%"""

            # Outcomes displayed using a label and f-string
            outcomes_text = Label(text=f"""
            ==================================
            Most likely class: {class_indexes[index[9]]}
            | Probability: {float('%.3f'%(probabilities[0, index[9]]))*100}%
        
            Second most likely class: {class_indexes[index[8]]} 
            | Probability: {float('%.3f'%(probabilities[0, index[8]]))*100}%
        
            Third most likely class: {class_indexes[index[7]]} 
            | Probability: {float('%.3f'%(probabilities[0, index[7]]))*100}%
        
            Fourth most likely class: {class_indexes[index[6]]} 
            | Probability: {float('%.3f'%(probabilities[0, index[6]]))*100}%""",
                                  pos=(250, 100))

            # Adds label to the display
            self.add_widget(outcomes_text)

            return class_indexes, probabilities, results

        except:
            error = "No image to be analysed |\n Please first import an image"
            open_message(error)

    # Saves the results or outcomes onto a text file
    def save_results(self):
        global results

        try:
            with open('result_file.txt', 'a') as f:
                f.write(results)

            prompt = "Results have been saved to a text file"
            open_message(prompt)

        except:
            error = "No results to be saved |\n Please first import an image"
            open_message(error)

def open_message(message):
    message_popup = Popup(title='Error Message', content=Label(text=message), size_hint=(None, None), size=(270, 270))
    message_popup.open()

# Builds the app using Kivy
class CNNApp(App):
    def build(self):
        self.title = "Python Image Classifier"
        return screen_manager

# Loads the .kv file (without the naming convention)
load_kv  = Builder.load_file("cnn.kv")

# Creates window manager (for the multiple screens)
screen_manager = RootScreen()

# Creates the different screens (for each page of the app)
screens = [StartWindow(name="start"), InfoWindow(name="info"), CNNWindow(name="cnn")]
# For loop for adding the screens/display
for screen in screens:
    screen_manager.add_widget(screen)

# Sets the opening screen/display
screen_manager.current = "start"

# Runs the application
if __name__ == "__main__":
    CNNApp().run()

