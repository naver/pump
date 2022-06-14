# Copyright 2022-present NAVER Corp.
# CC BY-NC-SA 4.0
# Available only for non-commercial use

import test_singlescale as ss
import test_singlescale_recursive as ssr
import test_multiscale as ms

def arg_parser():
    parser = ssr.arg_parser(ms.arg_parser())
    return parser

class Main (ms.Main):
    @staticmethod
    def build_matcher(args, device):
        # get a single-scale recursive matcher
        matcher = ssr.Main.build_matcher(args, device)
        type(matcher).demultiplex_img_trf = ms.demultiplex_img_trf # update transformer

        options = Main.get_options(args)
        return Main.tune_matcher(args, ms.MultiScalePUMP(matcher, **options), device).to(device)

if __name__ == '__main__':
    Main().run_from_args(arg_parser().parse_args())
