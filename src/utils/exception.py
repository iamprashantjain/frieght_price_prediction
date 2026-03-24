import sys

class customexception(Exception):
    def __init__(self, error_message, error_details: sys):
        super().__init__(error_message)
        self.error_message = error_message
        _, _, exc_tb = error_details.exc_info()
        
        if exc_tb:
            self.lineno = exc_tb.tb_lineno
            self.file_name = exc_tb.tb_frame.f_code.co_filename
        else:
            self.lineno = None
            self.file_name = None

    def __str__(self):
        if self.file_name and self.lineno:
            return "Error occurred in python script name [{0}] line number [{1}] error message [{2}]".format(
                self.file_name, self.lineno, str(self.error_message))
        else:
            return str(self.error_message)