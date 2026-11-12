"""Process entry points. The scheduler function is imported by name in every spawned rank."""


def run_scheduler_process(*args, **kwargs):
    from qwen_ttl import hooks
    hooks.install()
    from sglang.srt.managers.scheduler import run_scheduler_process as original
    return original(*args, **kwargs)


def main(argv):
    import os
    from sglang.srt.plugins import load_plugins
    from sglang.srt.server_args import prepare_server_args
    from sglang.srt.utils import kill_process_tree
    load_plugins()
    server_args = prepare_server_args(argv)
    try:
        server_args.resolve_once()
        from sglang.srt.entrypoints.http_server import app, launch_server
        from qwen_ttl import http
        http.install(app)
        launch_server(server_args, run_scheduler_process_func=run_scheduler_process)
    finally:
        kill_process_tree(os.getpid(), include_parent=False)
