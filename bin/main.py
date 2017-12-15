#!/usr/bin/env python
"""
coding=utf-8

Code Template

"""
import logging

import lib


def main():
    """
    Main function documentation template
    :return: None
    :rtype: None
    """
    logging.basicConfig(level=logging.DEBUG)

    extract()
    transform()
    model()
    load()
    pass


def extract():
    logging.info('Begin extract')

    # TODO Check if Niezsche is available. If not, download.

    # TODO Create observations DataFrame

    lib.archive_dataset_schemas('extract', locals(), globals())
    logging.info('End extract')
    pass


def transform():
    logging.info('Begin transform')

    # TODO Filter characters

    # TODO Create character ngrams

    # TODO Convert character ngrams into label representations

    # TODO Create y, containing next character after window

    lib.archive_dataset_schemas('transform', locals(), globals())
    logging.info('End transform')
    pass


def model():
    logging.info('Begin model')

    # TODO Create embedding

    # TODO Create model architecture

    # TODO Train model

    lib.archive_dataset_schemas('model', locals(), globals())
    logging.info('End model')
    pass


def load():
    logging.info('Begin load')

    # TODO Serialize encoder

    # TODO Serialize model



    lib.archive_dataset_schemas('load', locals(), globals())
    logging.info('End load')
    pass


# Main section
if __name__ == '__main__':
    main()
