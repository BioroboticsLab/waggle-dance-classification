import numpy as np

def classify_dance(img_array, model, img_count):
    """
    Uses the model of the neural network to give predictions for every single window of a dance.
    :param img_array: sequence to be classified
    :param model: model of the neural network
    """
    x_verify = []
    for i in range(img_array.shape[0]-img_count):
        x_verify.append(img_array[i:i+img_count,:,:])
    x_verify = np.asarray(x_verify)
    x_verify = x_verify.astype("float32")
    x_verify /= 255
    return model.predict(x_verify, batch_size=32, verbose=0)

