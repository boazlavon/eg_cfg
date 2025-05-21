from args_utils import get_grid_cmdline_args, generate_grid_configs
from eg_cfg import start_eg_cfg_session_manager
from consts import *


def main():
    args = get_grid_cmdline_args()
    session_config, inference_sessions_configs = generate_grid_configs(
        args.inference_session_grid_json, args.session_config_json
    )
    start_eg_cfg_session_manager(session_config, inference_sessions_configs)


if __name__ == "__main__":
    main()
