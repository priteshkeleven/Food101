from fastapi import FastAPI, File, UploadFile
import tensorflow as tf
from PIL import Image
import numpy as np
import io

# Initializing application
app = FastAPI()
loaded_model = None

# defining all classes that training dataset had
class_names = ["apple_pie", "baby_back_ribs", "baklava", "beef_carpaccio", "beef_tartare", "beet_salad", "beignets", "bibimbap", "bread_pudding", "breakfast_burrito", "bruschetta", "caesar_salad", "cannoli", "caprese_salad", "carrot_cake", "ceviche", "cheese_plate", "cheesecake", "chicken_curry", "chicken_quesadilla", "chicken_wings", "chocolate_cake", "chocolate_mousse", "churros", "clam_chowder", "club_sandwich", "crab_cakes", "creme_brulee", "croque_madame", "cup_cakes", "deviled_eggs", "donuts", "dumplings", "edamame", "eggs_benedict", "escargots", "falafel", "filet_mignon", "fish_and_chips", "foie_gras", "french_fries", "french_onion_soup", "french_toast", "fried_calamari", "fried_rice", "frozen_yogurt", "garlic_bread", "gnocchi",
               "greek_salad", "grilled_cheese_sandwich", "grilled_salmon", "guacamole", "gyoza", "hamburger", "hot_and_sour_soup", "hot_dog", "huevos_rancheros", "hummus", "ice_cream", "lasagna", "lobster_bisque", "lobster_roll_sandwich", "macaroni_and_cheese", "macarons", "miso_soup", "mussels", "nachos", "omelette", "onion_rings", "oysters", "pad_thai", "paella", "pancakes", "panna_cotta", "peking_duck", "pho", "pizza", "pork_chop", "poutine", "prime_rib", "pulled_pork_sandwich", "ramen", "ravioli", "red_velvet_cake", "risotto", "samosa", "sashimi", "scallops", "seaweed_salad", "shrimp_and_grits", "spaghetti_bolognese", "spaghetti_carbonara", "spring_rolls", "steak", "strawberry_shortcake", "sushi", "tacos", "takoyaki", "tiramisu", "tuna_tartare", "waffles"]


def load_and_prep_image(uploaded_file, img_size=224, scale=True):
    """
    reads in an image form filename, turns it into a tensor and reshapes into specified shape

    Args:
        filename (str): path to target image
        img_shape (int): height/width dimentions of target image size
        scale (bool): scale pixel values from 0-255 to  0-1 or not

    Returns:
        Image tensor of shape (img_shape, img_shape, 3)
    """
    # Decode image into tensor
    img = tf.convert_to_tensor(uploaded_file)

    # Resize the image
    img = tf.image.resize(img, [img_size, img_size])
    
    # Scaling
    if scale:
        # Rescale the image
        return img/255.
    else:
        return img


@app.on_event("startup")
def startup_event():
    """
    Function to define events that need to be initialized with application.

    Args:
        None
        
    Returns:
        None
    """
    global loaded_model
    
    # Loading saved Keras model
    loaded_model = tf.keras.models.load_model("model/fine_tuned_model_v1.h5")
    
    # Printing model summary
    print(loaded_model.summary())


def read_imagefile(file) -> Image.Image:
    """
    Converts uploaed image to Numpy array

    Args:
        file (FileObj): uploaded file ubject

    Returns:
        Numpy array of image
    """
    image = np.array(Image.open(io.BytesIO(file)))
    return image


@app.post("/classify")
async def classify_image(file: UploadFile = File()):
    """
    gets uploaded file and predicts it's class

    Args:
        file (File): uploaded file

    Returns:
        Dictionary containing predicted class and predicted probability
    """
    # Checking if uploaded file is in supported format or not
    extension = file.filename.split(".")[-1] in ("jpg", "jpeg", "png")
    if not extension:
        return {"error": "Image must be jpg or png format!"}
    
    # Reading uploaded image file and getting it as array
    image = read_imagefile(await file.read())
    
    # Preping an image for prediction
    img = load_and_prep_image(image, scale=False)
    
    # Expaning a dimention, because model was trained using batch processing
    img_expanded = tf.expand_dims(img, axis=0)
    
    # Getting prdiciton probability
    pred_prob = loaded_model.predict(img_expanded)
    
    # Getting prdiction class
    pred_class = class_names[pred_prob.argmax()]
    return {"prediction": pred_class, "prob": f"{pred_prob.max():.2f}"}
