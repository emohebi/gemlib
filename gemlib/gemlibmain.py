import gemlib.visualization.engine as eng
from gemlib.validation.utilities import _decode_dict
import gemlib.visualization.cmapsetup as colorsetup
from gemlib.validation.utilities import  _error, EXIT_FAILURE
import json
import os
import optparse
from gemlib.validation import utilities

def main(args=None):
    colorsetup.setup()
    parser = optparse.OptionParser()
    parser.add_option('-i', "--json-filepath",
                      help="Json file as configuration for the engine.")
    (options, args) = parser.parse_args(args=args)
    # config_path = r"C:\Users\em2945\Documents\gemLib\configs\test2.json"
    config_path = options.json_filepath
    if config_path is None:
        parser.print_help()
        return

    if not os.path.isfile(config_path):
        _error(EXIT_FAILURE, "input json '{0}' is missing".format(config_path))

    param_path = config_path
    if os.path.isfile(param_path):
        with open(param_path, mode='r') as f:
            param_path = f.read()
    input = json.loads(param_path, object_pairs_hook=_decode_dict)
    pip = eng.VisualizationEngine(None)
    pip.setup_pipeline_inputs(input)
    pip.run(input)
    # pip.discover()
    # pip.writetofile()
    utilities._info('process is finished!!!')


if __name__ == "__main__":
    main()


