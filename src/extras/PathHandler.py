from pathlib import Path

#Path handler Class for easier access and saving of data (only usable from files in subfolder of src like tests or similar)
class Pathhandler:
    def __init__(self):
        self.CUR_PATH = Path.cwd()
        self.PARENT_PATH = self.CUR_PATH.parent
        self.DATA_PATH = self.PARENT_PATH / "data"
        self.EXAMPLE_DATA_PATH = self.DATA_PATH / "example_data"
        self.CALIB_FOLDER =  self.EXAMPLE_DATA_PATH / "example_calibration"
        self.IMAGE_FOLDER = self.EXAMPLE_DATA_PATH / "example_images"
        self.SETTINGS_FOLDER = self.DATA_PATH / "settings"
        self.SAVE_FOLDER = self.EXAMPLE_DATA_PATH / "dest"

    def get_cwd(self):
        return self.CUR_PATH

    #Set SAVE_FOLDER relative to data
    def set_save_folder(self, save_folder):
        self.SAVE_FOLDER = self.DATA_PATH / save_folder

    #Set SAVE_FOLDER relative to data
    def set_image_folder(self, save_folder):
        self.IMAGE_FOLDER = self.DATA_PATH / save_folder

    #Set CALIBR_FOLDER relative to data
    def set_calibration_images_folder(self, save_folder):
        self.CALIB_FOLDER = self.DATA_PATH / save_folder

    #Set SETTINGS_FOLDER relative to data
    def set_settings_folder(self, save_folder):
        self.SETTINGS_FOLDER = self.DATA_PATH / save_folder

