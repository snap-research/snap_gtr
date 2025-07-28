"""
TODO: replace build decoder, activation to more generic format
"""

from termcolor import cprint


def build_decoder(config):
    decoder_type = config.pop('type')
    if decoder_type == "TriPlaneDecoder":
        from .triplane_decoder import TriplaneDecoder
        decoder = TriplaneDecoder(**config)
    else:
        raise NotImplementedError(f"No {decoder_type}")

    return decoder
