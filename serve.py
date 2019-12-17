#! /usr/bin/python3

import re
import os
import cgi
import base64
import mimetypes
from socket import gethostbyname, gethostname


from model import PredictModel, get_names_of_images, new_folder
from http.server import HTTPServer, BaseHTTPRequestHandler



IP = 'localhost'
PORT_NUMBER = 3000

SERVER_ADDRESS = (IP, PORT_NUMBER)


class Handler(BaseHTTPRequestHandler):
    def __init__(self, *args, directory=None, **kwargs):
        self.predict_model = PredictModel()
        self.search_entities = get_names_of_images()
        super().__init__(*args, **kwargs)

    CLIENT_FILES = {
        '/': 'client/index.html',
        '/index.html': 'client/index.html',
        '/style.css': 'client/style.css',
    }

    MEDIA_PATH = r'\/media\/(\w)+\/(\w|[А-Яа-я])+\.(png|jpg|jpeg)$'

    SIMILAR_IMAGES_TEMPLATE = '''\
        <!DOCTYPE html>
        <html lang="ru">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <meta http-equiv="X-UA-Compatible" content="ie=edge">
            <link href="https://fonts.googleapis.com/css?family=Oswald:300,400,500&display=swap" rel="stylesheet">
            <link rel="stylesheet" href="./style.css">
            <title>ImgRec</title>
        </head>
        <body>
            <h1>Similar Images</h1>
            <hr>
            <ul class="images-list">
                <li>
                    <img src="%(source_image)s" />
                    <span>source image \n distance: 0</span>
                </li>
                %(images)s
            </ul>
        </body>
        </html>
    '''

    def __set_headers(self, content_type, content_length):
        self.send_response(200)
        self.send_header('Content-type', content_type)
        self.send_header('Content-Length', content_length)
        self.end_headers()

    def __resolve_get_path(self):
        file_path = os.curdir + os.sep

        if self.path in self.CLIENT_FILES:
            return file_path + self.CLIENT_FILES[self.path]

        if re.match(self.MEDIA_PATH, self.path):
            return file_path + self.path

        return None

    def do_GET(self):
        file_path = self.__resolve_get_path()

        if not file_path:
            self.send_error(404, f'File Not Found: {self.path}')
            return

        try:
            content_type, _ = mimetypes.guess_type(file_path)

            with open(file_path, 'rb') as file_binary:
                file = file_binary.read()
                self.__set_headers(content_type, len(file))
                self.wfile.write(file)

        except IOError:
            self.send_error(404, f'File Not Found: {self.path}')

    def do_POST(self):
        if self.path == '/similar':
            form = cgi.FieldStorage(
                fp=self.rfile,
                headers=self.headers,
                environ={
                    'REQUEST_METHOD': 'POST',
                    'CONTENT_TYPE': self.headers['Content-Type'],
                }
            )

            image = form['image']
            image_data = image.file.read()
            image_mime, _ = mimetypes.guess_type(image.filename)
            image_base64 = str(base64.b64encode(image_data), 'utf-8')
            similar_images, distances = self.predict_model.search_nearest(self.search_entities, image_data)

            content = ( self.SIMILAR_IMAGES_TEMPLATE %
                {
                    'images': '\n'.join(map(
                        lambda img, dist: f'''<li><img src="media/{new_folder}/{img}" /><div>distance:{dist}</div></li>''',
                        similar_images, distances
                    )),
                    'source_image': f'data:{image_mime};charset=utf-8;base64, {image_base64}'
                }
            )

            body = content.encode('UTF-8', 'replace')
            self.__set_headers('text/html;charset=utf-8', len(body))

            self.wfile.write(body)


try:
    server = HTTPServer(SERVER_ADDRESS, Handler)
    print('Started httpserver on address ', SERVER_ADDRESS)

    server.serve_forever()

except KeyboardInterrupt:
    print('^C received, shutting down the web server')
    server.socket.close()