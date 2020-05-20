from DCGan import dcgan_class
import datetime
import logging
import logging.config
import sys
import yaml
import argparse
from server import run_server
from app import create_app

from IPython import embed
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

TRAIN = 'train'
SERVER= 'server'

def Logger(file_name):
    formatter = logging.Formatter(fmt='%(asctime)s %(module)s,line: %(lineno)d %(levelname)8s | %(message)s',
                                  datefmt='%Y/%m/%d %H:%M:%S') # %I:%M:%S %p AM|PM format
    logging.basicConfig(filename = '%s.log' %(file_name),format= '%(asctime)s %(module)s,line: %(lineno)d %'
                                                                 '(levelname)8s | %(message)s',
                                            datefmt='%Y/%m/%d %H:%M:%S', filemode = 'w', level = logging.INFO)
    log_obj = logging.getLogger()
    log_obj.setLevel(logging.WARNING)
    # log_obj = logging.getLogger().addHandler(logging.StreamHandler())

    # console printer
    screen_handler = logging.StreamHandler(stream=sys.stdout) #stream=sys.stdout is similar to normal print
    screen_handler.setFormatter(formatter)
    logging.getLogger().addHandler(screen_handler)

    log_obj.warning("Logger object created successfully..")
    return log_obj

def parse_args(args):
    """
    Parse command line parameters. Primary entry point is `etl`.
    Sub-parsers are used for each of it's specific commands.

    Args:
      args ([str]): command line parameters as list of strings

    Returns:
      :obj:`argparse.Namespace`: command line parameters namespace
    """

    parser = argparse.ArgumentParser(
        description='A Python interface for training a DCGAN.')
    subparsers = parser.add_subparsers(help='sub-command help', dest='command')

    #create the parser for the "cp" command (to copy files to HDFS via
    #command line)
    parser_create = subparsers.add_parser(
        TRAIN, help='Train command. Example use:\n'
                     '$ python runner.py train -y data.yaml '
                     ' This command train DCGAN , paramethers yaml file'   )

    parser_create.add_argument(
        '-y', '--yaml',
        dest='conf_file',
        type=str,
        help='YAML configuration file',
        required = True
    )
    parser_create.add_argument(
        '-e', '--epochs',
        dest='epochs',
        type=int,
        help='number of epochs',
        required = True
    )
    parser_create.add_argument(
        '-b', '--batch_size',
        dest='batch',
        type=int,
        help='batch size',
        required=True
    )
    # parser_pre = subparsers.add_parser(
    #     PREPROCESS, help='Preprocess bulk command. Example use:\n'
    #                  '$ python runner.py pre -y data.yaml '
    #                  ' This command preprocess files in input_pre directory yaml file'   )
    #
    # parser_pre.add_argument(
    #     '-y', '--yaml',
    #     dest='conf_file',
    #     type=str,
    #     help='YAML configuration file',
    #     required = True
    # )
    # parser_pre.add_argument(
    #     '-f', '--failed_blocks',
    #     dest='failed_files',
    #     type=str,
    #     help='file failed blocks',
    #     required = False
    # )
    #
    parser_ser = subparsers.add_parser(
        SERVER, help='server command. Example use:\n'
                     '$ python runner.py server -y data.yaml '
                     ' This command run the RESTFUL API'   )

    parser_ser.add_argument(
        '-y', '--yaml',
        dest='conf_file',
        type=str,
        help='YAML configuration file',
        required = True
    )
    parser_ser.add_argument(
        '-m', '--model_file',
        dest='model_file',
        type=str,
        help='h5 model file',
        required=True
    )
    return parser.parse_args(args)
def main(args):
    def load_yaml(file):
        with open(file, 'r', encoding="utf-8") as stream:
            # some versions of yaml we need Fullloader
            data = yaml.load(stream, Loader=yaml.FullLoader)

        return data

    args = parse_args(args)
    mode = args.command
    RELATIVE_YAML_FILENAME = args.conf_file
    conf_file = load_yaml(RELATIVE_YAML_FILENAME)


    if mode == TRAIN:
        epochs = args.epochs
        batch_size = args.batch
        log_dir = conf_file['log_dir']
        output_dir = conf_file['output_dir']
        pic_dir = conf_file['pics_dir']
        models_dir = conf_file['models_dir']
        # size of the latent space
        latent_dim = conf_file['latent_dim']

        time1 = datetime.datetime.now()
        timestampStr = time1.strftime("%H_%M_%S_%f_%b_%d_%Y")
        file_name = log_dir + '/dcgan_' + str(timestampStr)
        log_obj = Logger(file_name)
        log_obj.warning("Start process {} ".format(time1))
        log_obj.warning("Create DCGAN object: ")

        dcgan= dcgan_class(log_obj, output_dir, pic_dir, models_dir, latent_dim)
        # create the discriminator
        d_model = dcgan.define_discriminator(in_shape=(32, 32, 3))
        # create the generator
        g_model = dcgan.define_generator(dcgan.latent_dim)
        # create the gan
        gan_model = dcgan.define_gan(g_model, d_model)
        # load image data
        dataset = dcgan.load_real_samples()
        #print(dataset.shape)
        log_obj.warning(f"Train DCGAN model number epochs{epochs}: ")
        dcgan.train(g_model, d_model, gan_model, dataset,  dcgan.latent_dim, epochs, batch_size)
        #embed()
        return

    elif mode == SERVER:
        # create log object
        model_file = args.model_file
        log_dir = conf_file['log_dir']
        time1 = datetime.datetime.now()
        timestampStr = time1.strftime("%H_%M_%S_%f_%b_%d_%Y")
        file_name = log_dir + '/nlp_'+str(timestampStr)
        log_obj = Logger(file_name)
        app = create_app(conf_file, log_obj, model_file)
        run_server(app)
        return
    return

def run():
    """
    Entry point for console_scripts
    """
    main(sys.argv[1:])


if __name__ == '__main__':
    run()
