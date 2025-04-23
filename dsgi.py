from args_utils import get_cmdline_args, build_dsgi_session_manager_args
from dsgi_session_manager import DsgiSessionManager


def main():
    args = get_cmdline_args()
    dsgi_session_manager_args = build_dsgi_session_manager_args(args)
    dsgi_session_manager = DsgiSessionManager(**dsgi_session_manager_args)
    dsgi_session_manager.setup()
    dsgi_session_manager.solve()


if __name__ == "__main__":
    main()
