"""Command line utilities
"""
import argparse
from subprocess import call

try:
    from mpi4py import MPI
except ImportError:
    MPI = None

class Command:
    @staticmethod
    def print_call(cmd, prefix="python"):
        """Call 'cmd' and print the command in terminal.
        Arg:
            cmd (list) : command to run in terminal
        """
        cmd = prefix.split(" ") + cmd
        output = "| " + " ".join(cmd) + " |"
        margin = "-" * len(output)
        call(["echo", margin])
        call(["echo", output])
        call(["echo", margin])
        return call(cmd)

    @staticmethod
    def execute(func):
        def wrapper(self, **override):
            cmd = {}
            cmd.update(func(self))
            cmd.update({"--" + k: override[k] for k in override.keys()})
            # run the command
            run = []
            for key in cmd:
                if key.startswith("-"):
                    if type(cmd[key]) is dict:
                        for v in cmd[key].keys():
                            run.append(str(key))
                            run.append(str(v) + ":" + str(cmd[key][v]))
                    elif type(cmd[key]) is list:
                        for v in cmd[key]:
                            run.append(str(key))
                            run.append(str(v))
                    else:
                        run.append(str(key))
                        run.append(str(cmd[key]))
                else:
                    run.append(str(cmd[key]))
            return Command.print_call(run, list(cmd.keys())[0])

        return wrapper


class ArgParser:
    def __init__(self):
        self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        self.dump = not MPI.COMM_WORLD.Get_rank() if MPI != None else True

    def parse(self, args):
        known, unknown = self.parser.parse_known_args(args)
        self.known_dict = vars(known)
        self.unknown_dict = self._parse_unknown_args(unknown)

        if self.dump:
            print("\nKnown arguments:")
            print("-----------------------------------")
            print("{:<30}{:<30}".format("Option", "Value"))
            for key, value in self.known_dict.items():
                print("{:<30}{:<30}".format(str(key), str(value)))
            print("\nUnknown arguments:")
            print("-----------------------------------")
            print("{:<30}{:<30}".format("Option", "Value"))
            for key, value in self.unknown_dict.items():
                print("{:<30}{:<30}".format(str(key), str(value)))
            print("===================================")

    def get_dict(self):
        args = {}
        args.update(self.known_dict)
        args["unknown_params"] = self.unknown_dict
        return args

    def _parse_unknown_args(self, args):
        """
        Parse arguments not consumed by arg parser into a dicitonary
        """
        retval = {}
        preceded_by_key = False
        for arg in args:
            if arg.startswith("--"):
                if "=" in arg:
                    key = arg.split("=")[0][2:]
                    value = arg.split("=")[1]
                    retval[key] = value
                else:
                    key = arg[2:]
                    preceded_by_key = True
            elif preceded_by_key:
                retval[key] = arg
                preceded_by_key = False
        return retval


if __name__ == "__main__":

    # Check Command class
    Command.print_call(["Hello World!"], "echo")

    # Argument parser check
    arg_parser = ArgParser()
    arg_parser.parser.add_argument("--test", type=int, default=0)
    arg_parser.parse(["--check", "a", "--test=2"])
    print(arg_parser.get_dict())
