import argparse
from pythonosc import dispatcher
from pythonosc import osc_server

def print_message(address, *args):
    print(f"Received message on {address}: {args}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ip", default="127.0.0.1", help="The IP to listen on")
    parser.add_argument("--port", type=int, default=5005, help="The port to listen on")
    args = parser.parse_args()

    # Set up the dispatcher to handle messages
    disp = dispatcher.Dispatcher()
    disp.map("/randomNum", print_message)  # Map the message path to the print function

    # Create and start the server
    server = osc_server.ThreadingOSCUDPServer((args.ip, args.port), disp)
    print(f"Serving on {server.server_address}")
    server.serve_forever()
