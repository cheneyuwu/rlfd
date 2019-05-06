"""Command line utilities
"""

from subprocess import call


class Command:
    @staticmethod
    def print_call(cmd, prefix="python"):
        """Call 'cmd' and print the command in terminal.
        Arg:
            cmd (list) : command to run in terminal
        """
        cmd = [prefix] + cmd
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
                    run.append(str(key))
                run.append(str(cmd[key]))
            return Command.print_call(run)

        return wrapper


if __name__ == "__main__":
    Command.print_call(["Hello World!"], "echo")
