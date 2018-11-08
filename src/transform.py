import cv2
import glob
import copy
import os

from types import IntType, StringType
from Tkinter import Button, Label, Checkbutton, StringVar, Tk, S, W, N, E
from PIL import Image, ImageTk

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

    def __init__(self):
        self.snow = StringVar()         # S
        self.rain = StringVar()         # R
        self.storm = StringVar()        # T
        self.strong_wind = StringVar()  # W
        self.clouds = StringVar()       # C

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


class TrainingImage(object):
    """ 
    Class responsible for transformation of meteorogram image into labeled training example.
    """

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
        self._features = FeatureSet()

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

    def save_preview(self, destination_path):
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

    def save(self, output_dir):
        """
        Method saves labeled training example into a specified destination file.

        Args:
            output_dir (str): path to the directory where the training example should be stored.
        """

        assert type(output_dir) is StringType, 'output_dir: passed object of incorrect type'
        path = output_dir + self._filename.replace('.png', '@%s.png' % (self._features.label))
        cv2.imwrite(path, self._focus)

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

class FileProvider(object):
    """ 
    Class responsible for providing paths for unprocessed meteorograms.

    Args:
        preview_path (str): Full path where the temporary preview image is stored.
        total_files_count (int): Total number of files which need to be processed.
        current_file_index (int): Index of the first unprocessed file.
    """

    def __init__(self, input_dir, output_dir, preview_path):
        """
        Args:
            input_dir (str): Path to the directory which contains meteorograms.
            output_dir (int): Path to the directory where labeled training examples will be stored.
            preview_path (int): Path to a file where a temporary meteorogram preview will be stored.
        """

        assert type(input_dir) is StringType, 'input_dir: passed object of incorrect type'
        assert type(output_dir) is StringType, 'output_dir: passed object of incorrect type'
        assert type(preview_path) is StringType, 'preview_path: passed object of incorrect type'

        self._input_dir = input_dir           
        self._iterator = glob.iglob(self._input_dir+'*.png')

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        self.output_dir = output_dir
        self.preview_path = preview_path
        self.total_files_count = len(glob.glob(self._input_dir+'*.png'))
        self.current_file_index = 0
        assert self.total_files_count > 0, 'Not found any meteorograms to process'

    def next(self):
        """ 
        Returns:
            str: full path to the next unprocessed image.
        """
        path_processed = True
        while path_processed:
            self.current_file_index += 1
            next_path = self._iterator.next()
            path_processed = self._is_processed(next_path)            
        return next_path

    def _is_processed(self, path):
        """ 
        Method checks whether the requested file was already processed and move on to the next one if yes. 

        Args:
            path (str): full path to the meteorogram file.

        Returns:
            bool: True if the file was already processed, False otherwise.
        """

        pth = copy.copy(path)
        pth = pth.replace(self._input_dir, '')
        pth = pth.replace('.png', '')
        pth = self.output_dir + pth + '@*.png'
        exists = len(glob.glob(pth)) > 0
        return exists

class TrainingSetEditor(object):
    """ Class responsible for rendering editor's UI. """

    _crop_area = CropArea(65,140,180,466)
    _feature_labels = [('Snow', 'S'), ('Rain', 'R'), ('Storm', 'T'), ('Strong wind', 'W'), ('Clouds', 'C')]

    def __init__(self, size, file_provider):
        """
        Args:
            size (EditorSize): class representing internal sizes of the editor.
            file_provider (FileProvider): class providing files for processing.
        """
        assert type(size) is EditorSize, 'size: passed object of incorrect type'
        assert type(file_provider) is FileProvider, 'file_provider: passed object of incorrect type'

        self._size = size
        self._file_provider = file_provider
        self._current_path = self._file_provider.next()
        
    def _setup(self, size):
        """ Method which does the initial setup of the UI. """
        self._root_window = Tk()
        self._root_window.title("Training set editor")
        self._root_window.resizable(width=False, height=False)
        
        self._setup_progress(size)
        self._setup_image_label(size)
        self._setup_features(size)
        self._setup_next_button(size)

    def _setup_progress(self, size):
        """ Method which does the initial setup of progress labels. """
        fp = self._file_provider
        labels = ['Processing image', str(fp.current_file_index), 'out of', str(fp.total_files_count)]

        for index in range(len(labels)):
            label = Label(self._root_window, text=labels[index])
            label.grid(row=0, column=index)

        self._current_image_index_label = self._root_window.grid_slaves(row=0, column=1)[0]
        self._total_images_count_label = self._root_window.grid_slaves(row=0, column=len(labels)-1)[0]        

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
        self._next_button = Button(self._root_window, text="NEXT", command=self._on_next_button_click,)
        self._next_button.grid(row=3, column=0, columnspan=5, sticky=(N,W,E,S))
        self._root_window.bind('<Return>', self._on_next_button_click)            

    def _update_feature_buttons_binding(self):
        """ Method which rebinds keyboard shortcuts for the next meteorogram image. """
        self._buttons['S'].config(variable=self._current_image._features.snow)
        self._buttons['R'].config(variable=self._current_image._features.rain)
        self._buttons['T'].config(variable=self._current_image._features.storm)
        self._buttons['W'].config(variable=self._current_image._features.strong_wind)
        self._buttons['C'].config(variable=self._current_image._features.clouds)

    def _update_progess_labels(self):
        """ Method progress labels when a meteorogram was processed. """
        self._current_image_index_label.config(text=str(self._file_provider.current_file_index))
        self._total_images_count_label.config(text=str(self._file_provider.total_files_count))

    def _on_feature_toggled(self, event):
        """ Method called when feature checkbox was toggled. """
        self._buttons[event.char.upper()].toggle()

    def _on_next_button_click(self, event=None):
        """ Method called when next button was activated. """
        self._current_image.save(self._file_provider.output_dir)
        self._current_path = self._file_provider.next()
        self._show_image(self._current_path)
        self._update_progess_labels()

    def _show_image(self, path):
        """ Method displays the next meteorogram image. """
        self._current_image = TrainingImage(path, self._crop_area)
        self._current_image.save_preview(self._file_provider.preview_path)
        self._update_feature_buttons_binding()

        img = ImageTk.PhotoImage(Image.open(self._file_provider.preview_path))        
        self._label.configure(image=img)
        self._label.image = img        

    def activate(self):
        """ Method displays editor's window on the screen. """       
        self._setup(self._size)
        self._show_image(self._current_path)
        self._root_window.mainloop()

# Execution section
fileProvider = FileProvider('../source-images/', '../training-examples/', '../tmp-preview.png')
editor = TrainingSetEditor(EditorSize(630, 660), fileProvider)
editor.activate()