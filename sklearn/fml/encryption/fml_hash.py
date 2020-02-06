import hashlib

class FMLHash:
    def __init__(self):
        return
    
    def hashVal(self, value):
        m = hashlib.sha256()
        m.update(value.encode())
        return m.digest()
        
    def hashValAndReturnString(self, value):
        m = hashlib.sha1()
        m.update(value.encode())
        return m.digest().decode("unicode_escape")