import os
import os.path
from base64 import b64encode
from io import BytesIO

from werkzeug.utils import secure_filename

from flask import Flask, abort, render_template, request, url_for
from googleapiclient import discovery
from oauth2client.client import GoogleCredentials
from PIL import Image

app = application = Flask(__name__)


LABELS = '''✈️
🚗
🐦
🐱
🦄
🐶
🐸
🐴
🛳
🚚
'''.split()


def get_b64_image_bytes(file):
    im = Image.open(file)
    crop_size = min(im.size)
    crop_width = (im.width - crop_size) / 2
    crop_height = (im.height - crop_size) / 2
    crop_box = (crop_width, crop_height, crop_width + crop_size, crop_height +
                crop_size)
    im = im.crop(crop_box)
    im = im.resize((32, 32), resample=Image.LANCZOS)
    b = BytesIO()
    im.save(b, 'JPEG')
    image_bytes = b.getvalue()
    return b64encode(image_bytes).decode()


def predict(instance):
    credentials = GoogleCredentials.get_application_default()
    service = discovery.build('ml', 'v1', credentials=credentials)
    project_id = os.getenv('PROJECT_ID')
    model_name = os.getenv('MODEL_NAME')
    name = 'projects/{}/models/{}'.format(project_id, model_name)
    response = service.projects().predict(
      name=name,
      body={'instances': instance}
    ).execute()
    if 'error' in response:
      abort(500, response['error'])
    return response['predictions'][0]


@app.route('/', methods=['GET', 'POST'])
def index():
  context = {}
  if request.method == 'POST':
    image_file = request.files['image']
    if not image_file.mimetype.startswith('image'):
      abort(400)
    image_filename = secure_filename(image_file.filename)
    image_file.save(os.path.join('static', 'uploads', image_filename))
    image_url = url_for('static',
                        filename=os.path.join('uploads', image_filename))
    b64_image_bytes = get_b64_image_bytes(image_file.stream)
    instance = {'inputs': {'b64': b64_image_bytes}}
    label_index = predict(instance).get('outputs')
    label = LABELS[label_index]
    context.update({
      'image_url': image_url,
      'label': label or 'IDK',
    })
  return render_template('index.html', **context)


if __name__ == '__main__':
  app.run(debug=True)
