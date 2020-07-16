import json

def build_response(obj, meta_data):
    if not meta_data:
        del obj['meta_features']
    return obj

def construct_response(data, meta_data=False):
    response = None
    if type(data) == list:
        response = []
        for obj in data:
            response.append(build_response(obj, meta_data))
    else:
        response = build_response(data, meta_data)
            
    return response