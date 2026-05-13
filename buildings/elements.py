

class Element:
    def __init__(self, class_id, bbox, score, material):
        self.class_id = class_id
        self.bbox = bbox
        self.score = score
        self.material = None
        
        
        #call MC.predict(material)