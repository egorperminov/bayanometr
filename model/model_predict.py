#! /usr/bin/python3
from keras.applications import VGG19
from keras.applications.vgg19 import preprocess_input
from keras.preprocessing import image
from keras.engine import Model
from keras import backend as K
import io
from PIL import Image

from sklearn.neighbors import NearestNeighbors
from .get_files import *


class PredictModel:

    def vgg19_init(self):
        self.bm = VGG19(weights='imagenet')
        self.model = Model(inputs=self.bm.input, outputs=self.bm.get_layer('fc1').output)

    def knn_init(self):
        self.knn = NearestNeighbors(metric='cosine', algorithm='brute')

    def prep_database_files(self, arr_data_names):
        vectors = []
        for file_name in arr_data_names:
            path = join(path_data_images, file_name)
            img = image.load_img(path, target_size=(224, 224))
            x = image.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            x = preprocess_input(x)
            vector = self.model.predict(x).ravel()
            print(f'File {file_name} converted to vector')
            vectors.append(vector)
        return vectors

    def prep_input_file(self, image_binary):
        image_object = Image.open(io.BytesIO(image_binary))
        resized_image = image_object.resize((224, 224))
        x = image.img_to_array(resized_image)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        vector = self.model.predict(x).ravel()
        vector = np.array(vector, np.float32).reshape(1, -1)
        print(f'User file converted to array')
        return vector

    def get_files_names_and_vectors(self, arr_data_names):
        if json_file_exist():
            return get_predicted_vectors_from_json()
        database_vectors = self.prep_database_files(arr_data_names)
        make_json_file(arr_data_names, database_vectors)
        return get_predicted_vectors_from_json()

    def search_nearest(self, arr_data_names, input_binary_file):
        # Initialize models
        self.vgg19_init()
        self.knn_init()
        # Get vectors of images
        input_vector = self.prep_input_file(input_binary_file)
        files_names, database_vectors = self.get_files_names_and_vectors(arr_data_names)
        # Fitting model
        self.knn.fit(database_vectors)
        # Search similar images
        dist, indices = self.knn.kneighbors(input_vector, n_neighbors=10)
        names_similar_images, dists = zip(*[(files_names[indices[0][i]], dist[0][i]) for i in range(len(indices[0]))])
        # End model session
        K.clear_session()
        return names_similar_images, dists


