class_to_lbl = {
    #define here the data classes ex.:
    # 0: "pen",
    # 1: "microphone",
    # ...
    0: "pen",
    1: "shoe",
    2: "plant",
    3: "table",
    4: "knife",
    5: "glass",
    6: "cat",
    7: "cherry",
}

train_annotation = {
    #define here the train data annotations:
    # "xxx.jpg": 4,
    # "xxy.jpg": 3,
    # ...
    "004.jpg": 0, #  IMAGE= 00!:JPG BELONGS TO CLASS PEN
    "005.jpg": 1, #  IMAGE= 00!:JPG BELONGS TO CLASS SHOE
    "007.jpg": 1, #  IMAGE= 00!:JPG BELONGS TO CLASS SHOE
    "008.jpg": 2, #  IMAGE= 00!:JPG BELONGS TO CLASS PLANT
    "009.jpg": 3,  #  IMAGE= 00!:JPG BELONGS TO CLASS TABLE
    "011.jpg": 0,
    "013.jpg": 4,
    "016.jpg": 4,
    "017.jpg": 5,
    "019.jpg": 6,
    "020.jpg": 7,
    }

val_annotation = {
    #define here the val data annotations:
    # "zzz.jpg": 0,
    # "zzw.jpg": 7,
    # ...
    "000.jpg": 0, #  IMAGE= 00!:JPG BELONGS TO CLASS PEN
    "006.jpg": 1, #  IMAGE= 00!:JPG BELONGS TO CLASS SHOE 
    "012.jpg": 2, #  IMAGE= 00!:JPG BELONGS TO CLASS PLANT
    "034.jpg": 3, #  IMAGE= 00!:JPG BELONGS TO CLASS TABLE
    "029.jpg": 4, 
    "049.jpg": 5,
    "031.jpg": 6,
    "050.jpg": 7,}