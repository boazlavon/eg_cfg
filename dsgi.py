from args_utils import get_cmdline_args, build_dsgi_session_manager_config
from dsgi_session_manager import DsgiSessionManager


def main():
    cmdline_args = get_cmdline_args()
    session_config, inference_session_config = build_dsgi_session_manager_config(
        cmdline_args
    )
    inference_sessions_configs = [inference_session_config]
    dsgi_session_manager = DsgiSessionManager(
        session_config, inference_sessions_configs
    )
    dsgi_session_manager.setup()
    dsgi_session_manager.solve()


if __name__ == "__main__":
    main()
