from args_utils import get_cmdline_args, build_session_config
from dsgi_session_manager import DsgiSessionManager
import pprint


def start_dsgi_session_manager(session_config, inference_sessions_configs):
    print(f"Generated {len(inference_sessions_configs)} configurations.")
    pprint.pprint(session_config)
    pprint.pprint(inference_sessions_configs)
    dsgi_session_manager = DsgiSessionManager(
        session_config, inference_sessions_configs
    )
    print("DSGI Session manager Setup started")
    dsgi_session_manager.setup()
    dsgi_session_manager.solve()


def main():
    cmdline_args = get_cmdline_args()
    session_config, inference_session_config = build_session_config(cmdline_args)
    start_dsgi_session_manager(session_config, [inference_session_config])


if __name__ == "__main__":
    main()
