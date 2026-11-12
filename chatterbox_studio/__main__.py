import argparse


def main():
    ap = argparse.ArgumentParser(prog="chatterbox_studio")
    ap.add_argument("mode", choices=["gui", "serve", "mcp"])
    ap.add_argument("--device", default=None, help="cuda | cpu | mps (default: auto)")
    args = ap.parse_args()

    if args.mode == "mcp":
        from .mcp_server import main as run_mcp
        run_mcp()
    elif args.mode == "serve":
        import uvicorn
        from . import config
        from .api import create_app
        from .engine import TTSEngine
        from .voices import VoiceStore
        app = create_app(TTSEngine(args.device), VoiceStore())
        uvicorn.run(app, host=config.API_HOST, port=config.API_PORT)
    else:
        from .gui import run_gui
        run_gui(args.device)


main()
