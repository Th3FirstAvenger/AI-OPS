"""Tool base class"""
import json
import subprocess


class Tool:
    """Represents a Penetration Testing Tool"""
    name: str
    tool_description: str
    args_description: str
    use_case: str = ''

    def __init__(
            self,
            name: str,
            tool_description: str,
            args_description: str,
            use_case: str
    ):
        self.name = name
        self.tool_description = tool_description
        self.args_description = args_description
        self.use_case = use_case

    @staticmethod
    def load_tool(path: str):
        """Get tool description from json file"""
        # TODO: refactor schema validation in another method
        with open(path, 'r', encoding='utf-8') as fp:
            tool_data: dict = json.load(fp)
            keys = ['name', 'tool_description', 'args_description', 'use_case']

            if not isinstance(tool_data, dict):
                raise TypeError(
                    f"Wrong format for {path}.\n"
                    f"Expected: dict\n"
                    f"Got: {type(tool_data)}."
                )

            valid_keys = False in [key in keys for key in tool_data.keys()]
            if len(tool_data) != 4 or valid_keys:
                raise ValueError(f"Wrong format at {path}: invalid keys.")

            name = tool_data['name']
            tool_description = ''.join(tool_data['tool_description'])
            # in case args_description is provided as string instead
            # of string list it still works: ''.join(string) returns string
            args_description = ''.join(tool_data['args_description'])
            use_case = tool_data['use_case']

            if not (name and tool_description and args_description):
                raise ValueError(f"Wrong format at {path}\nFound empty values")

            return Tool(name, tool_description, args_description, use_case)

    @staticmethod
    def run(*args):
        """Execute a tool"""
        if not isinstance(args[0], str):
            raise TypeError(f'Argument must be a string found {type(args[0])}.')
        if len(args[0]) == 0:
            raise ValueError('String is empty.')

        command = args[0].encode('utf-8').decode('utf-8')
        arguments = command.split()
        result = subprocess.run(arguments, capture_output=True, check=False)

        stdout = result.stdout.decode('utf-8', errors='replace')
        stderr = result.stderr.decode('utf-8', errors='replace')

        if len(stderr) > 0:
            return f'{stdout}\n{stderr}'
        return stdout

    def get_documentation(self):
        """Used to provide documentation for the Agent LLM"""
        return f"""
Tool: {self.name}
Description: 
    {self.tool_description}
Arguments:
    {self.args_description}          
"""
