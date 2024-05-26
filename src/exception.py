import sys

from src.logger import logging

def error_message_detail(error,error_detail :sys):
    _,_,exc_tb = error_detail.exc_info()
    file_name = exc_tb.tb_frame.f_code.co_filename
    lineno = exc_tb.tb_lineno
    error_message = f"Error occured in python script name {file_name} line number {lineno} error_message {str(error)}"
    return error_message

class CustomException(Exception):
    def __init__(self,error,error_detail:sys):
        self.error_message = error_message_detail(error,error_detail)

    def __str__(self) -> str:
        return self.error_message