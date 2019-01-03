import glob
import json
import os

from types import IntType, StringType

class TrainingSetBuilder(object):
    """
    Class which is responsible for managing index of the training set.
    """

    index = { "sorted_keys": [], "values": {} , "active_index": 0 }

    def load_index(self, input_file):
        """
        Method which loads training set index from file.

        Args:
            input_file (StringVar): path to the input file.
        """

        self._validate_path(input_file)

        with open(input_file) as infile:
            self.index = json.load(infile)

    def scan_input(self, input_dir):
        """
        Method which scans the directory with example images and builds 
        a sorted list of examples names.

        Args:
            input_dir (StringVar): path to the input directory.
        """

        self._validate_path(input_dir)

        for path in glob.iglob(input_dir+'*.png'):
            base_filename = os.path.splitext(os.path.basename(path))[0]
            self.index["sorted_keys"].append(base_filename)

    def scan_values(self, input_dir):        
        self._validate_path(input_dir)

        for path in glob.iglob(input_dir+'*.png'):
            base_filename = os.path.splitext(os.path.basename(path))[0].split('@')            
            key = base_filename[0]
            value = base_filename[1]            
            self.index["values"][key] = value

    def save(self, output_path):
        """
        Method which dumps the index to the specified file.

        Args:
            output_path (StringVar): path to the input directory.        
        """

        assert type(output_path) is StringType, 'output_path: passed object of incorrect type'

        with open(output_path, 'w') as outfile:
            json.dump(self.index, outfile, indent=4, separators=(',', ':'))

    def _validate_path(self, path):
        assert type(path) is StringType, 'path: passed object of incorrect type'

        if not os.path.exists(path):
            raise ValueError('File or directory %s does not exists' % (path))        

# Execution section
builder = TrainingSetBuilder()
builder.scan_input('../source-images/')
builder.scan_values('../training-examples/')
builder.save('../training-set-index.json')

# builder.load_index('../training-set-index.json')
# print(builder.index['values']['2017122006-432-277'])
