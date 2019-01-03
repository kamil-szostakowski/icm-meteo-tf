import cv2
import glob
import copy
import os
import json

from types import IntType, StringType
from Tkinter import Button, Label, Checkbutton, Text, StringVar, Tk, S, W, N, E, END, CENTER
from PIL import Image, ImageTk

class FeatureSet(object):
    """
    Class which stores information about all the features of the meteorogram.
    Attributes of that class should contain the following values if the appropriate feature was detected.

    Args:
        snow (StringVar): S if detected, empty string otherwise.
        rain (StringVar): R if detected, empty string otherwise.
        storm (StringVar): R if detected, empty string otherwise.
        strong_wind (StringVar): W if detected, empty string otherwise.
        clouds (StringVar): C if detected, empty string otherwise.
    """

    def __init__(self, features_repr=''):
        """
        Args:
            features_repr (StringType): String representing features detected on the meteorogram.
        """

        assert type(features_repr) is StringType, 'features_repr: passed object of incorrect type'
        
        self.snow = StringVar(value='S' if 'S' in features_repr else '')         # S
        self.rain = StringVar(value='R' if 'R' in features_repr else '')         # R
        self.storm = StringVar(value='T' if 'T' in features_repr else '')        # T
        self.strong_wind = StringVar(value='W' if 'W' in features_repr else '')  # W
        self.clouds = StringVar(value='C' if 'C' in features_repr else '')       # C

    @property
    def label(self):
        """ 
        Composes a label for a training example using set of detected features.
        Label can consists of any subset of uppercase characters (S,R,T,W,C) or ONLY U if no features were detected.

        Returns:
            string: label for a training example
        """
        lbl = ''.join([self.snow.get(), self.rain.get(), self.storm.get(), self.strong_wind.get(), self.clouds.get()])
        return lbl if len(lbl) > 0 else 'U'

class TrainingInput(object):
    """
    Class which represents a categorizable meteorogram image.

    Args:
        path (StringVar): Path to the meteorogram image.
        features (FeatureSet): Features detected on the specified meteorogram.
    """

    def __init__(self, path, features_repr):
        """
        Args:
            path (StringVar): Path to the meteorogram image.
            features_repr (StringType): String representing features detected on the meteorogram.
        """

        assert type(path) is StringType, 'path: passed object of incorrect type'
        assert type(features_repr) is StringType, 'features_repr: passed object of incorrect type'

        self.path = path
        self.features = FeatureSet(features_repr)

class TrainingDataStore(object):    
    """ 
    Class responsible for providing paths for unprocessed meteorograms.

    Args:
        preview_path (str): Full path where the temporary preview image is stored.
        total_files_count (int): Total number of files which need to be processed.
        current_file_index (int): Index of the first unprocessed file.
    """

    _index = { "sorted_keys": [], "values": {} }

    def __init__(self, input_dir, index_path, preview_path):
        """
        Args:
            input_dir (str): Path to the directory which contains meteorograms.
            index_path (int): Path to the json file which stores categories assigned to each input.
            preview_path (int): Path to a file where a temporary meteorogram preview will be stored.
        """

        assert type(input_dir) is StringType, 'input_dir: passed object of incorrect type'
        assert type(index_path) is StringType, 'index_path: passed object of incorrect type'
        assert type(preview_path) is StringType, 'preview_path: passed object of incorrect type'

        self._input_dir = input_dir
        self._index_path = index_path
        self.preview_path = preview_path        

        # Load index from file
        if os.path.exists(index_path):
            with open(index_path) as infile:
                self._index = json.load(infile)

        # Scan input_dir if index is empty 
        if len(self._index["sorted_keys"]) == 0:
            self._scan_input_dir(self._input_dir)
            self.dump_index()

        # Move to a first unprocessed image
        self.current_file_index = self._get_first_unprocessed_index()
        assert self.total_files_count > 0, 'Not found any meteorograms to process'

    @property
    def total_files_count(self):
        """
        Returns:
            int: total number of meteorogram images in the input_dir.
        """
        return len(self._index["sorted_keys"])

    def get_training_input(self, index):
        """
        Method which returns a training input from the specified index.
        If index is greater than the total_files_count, the last item will be returned.
        """
        assert type(index) is IntType

        self.current_file_index = min(index, len(self._index["sorted_keys"])-1)
        current_item = self._index["sorted_keys"][self.current_file_index].encode('ascii','ignore')
        current_item_path = os.path.join(self._input_dir, current_item + ".png")
        features = '' if not current_item in self._index["values"] else self._index["values"][current_item]

        return TrainingInput(current_item_path, features.encode('ascii','ignore'))

    def update_training_input(self, training_input):
        """
        Method updates the index with the training_input passed as a parameter in the index.
        If the training input is not in the index already, new entry will be created.

        Args:
            training_input (StringType): Path to the example image.
        """
        assert type(training_input) is TrainingInput, 'training_input: passed object of incorrect type'        

        filename = os.path.splitext(os.path.basename(training_input.path))[0]
        self._index["values"][filename] = training_input.features.label        

    def dump_index(self):
        """ Method which saves the current state of the index to file """
        with open(self._index_path, 'w') as outfile:
            json.dump(self._index, outfile, indent=4, separators=(',', ':'))
            print("Index dumped")

    def _get_first_unprocessed_index(self):
        """ Method returns an index of first unprocessed meteorogram image """
        index = 0
        for training_item in self._index["sorted_keys"]:
            if not training_item in self._index["values"]: # is already processed?
                break                
            index += 1
        return index        

    def _scan_input_dir(self, input_dir):
        """ Method scans input_dir and puts names of all meteorogram images to the index """
        for path in glob.iglob(input_dir+'*.png'):            
            filename = os.path.splitext(os.path.basename(path))[0]
            self._index["sorted_keys"].append(filename)

class CropArea(object): 
    """
    Class which encapsulates boundaries of the area which should be croped out
    from the meteorogram as an input for machine learning.

    Args:
        x (int): X coordinate of top left corner of crop rectangle (in pixels)
        y (int): Y coordinate of top left corner of crop rectangle (in pixels)
        width (int): width of the crop rectangle (in pixels)
        height (int): height of the crop rectangle (in pixels)
    """

    def __init__(self, x, y, width, height):        
        assert type(x) is IntType, 'x: passed object of incorrect type'
        assert type(y) is IntType, 'y: passed object of incorrect type'
        assert type(width) is IntType, 'width: passed object of incorrect type'
        assert type(height) is IntType, 'height: passed object of incorrect type'

        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.outline = 0

    @property
    def outline(self):
        """
        Returns:
            int: Width of crop rectangle's outline. 
        """
        return self._outline

    @outline.setter
    def outline(self, width):
        """ 
        Method sets the width of an outline.
        Outline is added outside of the defined area. It means that the pice of image
        which should be initially cropped will remain unchanged.

        - Outline extends CropArea's width and height.
        - Outline alters X and Y coordinates of crop rectangle.        

        Args:
            width (int): desired width of crop rectangle's outline (in pixels)
        """
        
        assert type(width) is IntType, 'width: passed object of incorrect type'
        self._outline = width
        self.x_slice = slice(self.x-width, self.x+self.width+width)
        self.y_slice = slice(self.y-width,self.y+self.height+width)              

    @property
    def dup(self):
        """ 
        Returns:
            CropArea: Copy of a CropArea object.
        """
        return copy.deepcopy(self)

class TrainingImagePreview(object):
    def __init__(self, img_path, crop):
        """
        Args:
            img_path (str): full path to the full meteorogram image.
            crop (CropArea): area which should be cropped out of the meteorogram in order to prepare a training example.
        """
        
        assert type(img_path) is StringType, 'img_path: passed object of incorrect type'
        assert type(crop) is CropArea, 'crop: passed object of incorrect type'

        oCrop = crop.dup
        oCrop.outline = 3
        
        self._filename = img_path.split('/')[-1]

        self.preview = cv2.imread(img_path)        
        self._focus = self.preview[crop.y_slice, crop.x_slice]
        self.preview = cv2.cvtColor(self.preview, cv2.COLOR_BGR2GRAY)
        self.preview = cv2.cvtColor(self.preview, cv2.COLOR_GRAY2BGR)                                
        self.preview[oCrop.y_slice, oCrop.x_slice] = self._outlined_focus(self._focus, oCrop.outline)  

    def _outlined_focus(self, focus, outline):
        """
        Returns an image which is then used for representation of the focused area on a meteorogram preview.

        Args:
            focus: area which should be cropped out as a treining example
            outline: width of an outline

        Returns:
            ndarray: Image representing cropped area with an outline.
        """
        return cv2.copyMakeBorder(focus, outline, outline, outline, outline, cv2.BORDER_CONSTANT, value=[0,0,255])

    def save(self, destination_path):
        """
        Method saves full preview image into a specified destination file.
        Preview image consists of:

        - Full grayscale meteorogram.
        - Color area which will be used as a training example.
        - Black outline around the focus area.

        Args:
            destination_path (str): full path where the temporary preview image should be stored.
        """

        assert type(destination_path) is StringType, 'destination_path: passed object of incorrect type'
        cv2.imwrite(destination_path, self.preview)

class EditorSize(object):
    """ Class which encapsulates internal sizes of editor's UI. """

    def __init__(self, width, height):
        """
        Args:
            width (int): max width of meteorogram image (in pixels).
            height (int): height of meteorogram image (in pixels).
        """

        assert type(width) is IntType, 'width: passed object of incorrect type'
        assert type(height) is IntType, 'height: passed object of incorrect type'
        self.content_width = width
        self.content_height = height

class TrainingSetEditor(object):
    """ Class responsible for rendering editor's UI. """

    _crop_area = CropArea(65,140,180,466)
    _feature_labels = [('Snow', 'S'), ('Rain', 'R'), ('Storm', 'T'), ('Strong wind', 'W'), ('Clouds', 'C')]

    def __init__(self, size, data_store):
        """
        Args:
            size (EditorSize): class representing internal sizes of the editor.
            data_store (TrainingDataStore): class providing files for processing.
        """
        assert type(size) is EditorSize, 'size: passed object of incorrect type'
        assert type(data_store) is TrainingDataStore, 'data_store: passed object of incorrect type'

        self._size = size
        self._data_store = data_store        
        
    def _setup(self, size):
        """ Method which does the initial setup of the UI. """
        self._root_window = Tk()
        self._root_window.title("Training set editor")
        self._root_window.resizable(width=False, height=False)
        
        self._setup_progress(size)
        self._setup_image_label(size)
        self._setup_features(size)
        self._setup_next_button(size)
        self._current_training_input = self._data_store.get_training_input(self._data_store.current_file_index)

    def _setup_progress(self, size):
        """ Method which does the initial setup of progress labels. """        
        label_1 = Label(self._root_window, text='Processing image')
        label_1.grid(row=0, column=0)

        self._current_image_index_text = Text(self._root_window, height=1, width=6)        
        self._current_image_index_text.insert(END, str(self._data_store.current_file_index))
        self._current_image_index_text.grid(row=0, column=1)

        label_3 = Label(self._root_window, text='out of')
        label_3.grid(row=0, column=2)

        self._total_images_count_label = Label(self._root_window, text=str(self._data_store.total_files_count))
        self._total_images_count_label.grid(row=0, column=3)                        

    def _setup_image_label(self, size):
        """ Method which does the initial setup of meteorogram preview image. """
        self._label = Label(self._root_window, width=size.content_width)
        self._label.grid(row=1, column=0, columnspan=5, sticky=(N,W,E,S))

    def _setup_features(self, size):
        """ Method which does the initial setup of features checkboxes. """
        self._buttons = {}
        for index in range(len(self._feature_labels)):
            label = self._feature_labels[index][0]
            value = self._feature_labels[index][1]
            checkbox = Checkbutton(self._root_window, text=label, onvalue=value, offvalue='')
            checkbox.grid(row=2, column=index)

            self._root_window.bind(value.lower(), self._on_feature_toggled)
            self._buttons[value] = checkbox

    def _setup_next_button(self, size):
        """ Method which does the initial setup of the next button. """
        self._next_button = Button(self._root_window, text='NEXT', command=self._on_next_button_click)
        self._next_button.grid(row=3, column=0, columnspan=5, sticky=(N,W,E,S))
        self._root_window.bind('<Return>', self._on_next_button_click)            

    def _update_feature_buttons_binding(self):
        """ Method which rebinds keyboard shortcuts for the next meteorogram image. """
        self._buttons['S'].config(variable=self._current_training_input.features.snow)
        self._buttons['R'].config(variable=self._current_training_input.features.rain)
        self._buttons['T'].config(variable=self._current_training_input.features.storm)
        self._buttons['W'].config(variable=self._current_training_input.features.strong_wind)
        self._buttons['C'].config(variable=self._current_training_input.features.clouds)

        for feature_sign in self._current_training_input.features.label:
            if feature_sign in self._buttons:
                self._buttons[feature_sign].select()                                               

    def _update_progess_labels(self):
        """ Method progress labels when a meteorogram was processed. """
        self._current_image_index_text.delete(1.0, END)
        self._current_image_index_text.insert(END, str(self._data_store.current_file_index))
        self._total_images_count_label.config(text=str(self._data_store.total_files_count))

    def _on_feature_toggled(self, event):
        """ Method called when feature checkbox was toggled. """
        self._buttons[event.char.upper()].toggle()

    def _on_next_button_click(self, event=None):
        """ Method called when next button was activated. """
        self._data_store.update_training_input(self._current_training_input)

        if self._data_store.current_file_index % 10 == 0:
            self._data_store.dump_index()
        
        try:
            self._current_training_input = self._data_store.get_training_input(self._get_next_item_index())
            self._show_image(self._current_training_input)
            self._update_progess_labels()
        except ValueError:
            self._update_progess_labels()

    def _get_next_item_index(self):
        """ Method returns the index of next meteorogram to process """
        newIndex = int(self._current_image_index_text.get(1.0, END).encode('ascii','ignore'))
        return newIndex if newIndex != self._data_store.current_file_index else newIndex+1

    def _show_image(self, training_input):
        """ Method displays the next meteorogram image. """
        self._current_preview = TrainingImagePreview(training_input.path, self._crop_area)
        self._current_preview.save(self._data_store.preview_path)        
        self._update_feature_buttons_binding()

        img = ImageTk.PhotoImage(Image.open(self._data_store.preview_path))        
        self._label.configure(image=img)
        self._label.image = img        

    def activate(self):
        """ Method displays editor's window on the screen. """       
        self._setup(self._size)
        self._show_image(self._current_training_input)
        self._root_window.mainloop()

# Execution section
dataStore = TrainingDataStore('../source-images/', '../training-set-index.json', '../tmp-preview.png')
editor = TrainingSetEditor(EditorSize(630, 660), dataStore)
editor.activate()