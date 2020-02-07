import hashlib

class FMLHash:
    def __init__(self):
        return
    
    def hashVal(self, value):
        """
        converts the value to a byte object and then hashes it
        """
        m = hashlib.sha256()
        m.update(value.encode())
        return m.digest()
        
    def hashValAndReturnString(self, value):
        """
        converts the value to a byte object and then hashes it and converts it to string
        """
        m = hashlib.sha1()
        m.update(value.encode())
        return m.digest().decode("unicode_escape")