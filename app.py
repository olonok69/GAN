from flask import Blueprint,Flask, send_file
from PIL import Image
main = Blueprint('main', __name__)
import io
import datetime
from DCGan import dcgan_class
from IPython import embed

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@main.route("/dcgan/predict", methods=["POST", "GET"])
def bulk_runtimea():
    b1= datetime.datetime.now().timestamp()
    X =  estimator_prob.predict()
    img=Image.fromarray(X[0, :, :], 'RGB')
    # create file-object in memory
    file_object = io.BytesIO()
    # write PNG in file-object
    img.save(file_object, 'PNG')
    # move to beginning of file so `send_file()` it will read from start
    file_object.seek(0)
    b2=datetime.datetime.now().timestamp()
    timeb=b2-b1
    logger.info("Request time ...: {}".format(timeb))
    return send_file(file_object, mimetype='image/PNG')

def create_app( conf_file, log_obj, model_file):

    global estimator_prob

    object=_create_objects(conf_file, log_obj,model_file)
    estimator_prob = object

    app = Flask(__name__)
    app.register_blueprint(main)
    return app


def _create_objects(conf_file, log_obj, model_file):
    output_dir = conf_file['output_dir']
    pic_dir = conf_file['pics_dir']
    models_dir = conf_file['models_dir']
    # size of the latent space
    latent_dim = conf_file['latent_dim']
    dcgan = dcgan_class(log_obj, output_dir, pic_dir, models_dir, latent_dim)
    model = dcgan.load_model(model_file)

    return dcgan