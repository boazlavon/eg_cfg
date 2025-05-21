from args_utils import get_cmdline_args, build_session_config
from eg_cfg_session_manager import EgCfgSessionManager
import pprint


def start_eg_cfg_session_manager(session_config, inference_sessions_configs):
    print(f"Generated {len(inference_sessions_configs)} configurations.")
    pprint.pprint(session_config)
    pprint.pprint(inference_sessions_configs)
    eg_cfg_session_manager = EgCfgSessionManager(
        session_config, inference_sessions_configs
    )
    print("EG_CFG Session manager Setup started")
    eg_cfg_session_manager.setup()
    eg_cfg_session_manager.solve()


def main():
    cmdline_args = get_cmdline_args()
    cmdline_args = vars(cmdline_args)
    session_config, inference_session_config = build_session_config(cmdline_args)
    start_eg_cfg_session_manager(session_config, [inference_session_config])


if __name__ == "__main__":
    main()
