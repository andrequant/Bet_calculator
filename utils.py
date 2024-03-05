def convert_to_float(input_string):
    if input_string is not None:
        try:
            # Replace comma with dot and attempt to convert to float
            return float(input_string.replace(',', '.'))
        except ValueError:
            # Return None if there is nothing to convert
            return None
    
def dict_to_array(data):
    return [list(inner_dict.values()) for inner_dict in data.values()]