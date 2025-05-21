from args_utils import get_trials_cmdline_args, convert_trials_to_configs
from eg_cfg import start_eg_cfg_session_manager
from consts import *


def main():
    args = get_trials_cmdline_args()
    session_config, inference_sessions_configs = convert_trials_to_configs(
        args.trials_json, args.session_config_json
    )
    start_eg_cfg_session_manager(session_config, inference_sessions_configs)


if __name__ == "__main__":
    main()
