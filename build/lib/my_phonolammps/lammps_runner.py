"""A wrapper for running generic lammps scripts with injected variables"""
from typing import List, Set
from enum import Enum
from dataclasses import dataclass
import re
import os

from my_phonolammps._lammps import MyLammps
from my_phonolammps._phonolammps import MyPhonolammps


class VarType(Enum):
    """enum of LAMMPS variable declaration types"""
    STRING = "string"
    EQUAL = "equal"


class MissingVarError(Exception):
    """raised when insufficient arguments passed for undefined LAMMPS input variables"""


@dataclass(frozen=True)
class LammpsVarLine:
    """
    Represents a LAMMPS command line of the sort:
        variable varname equal 123.321 or variable string file_name "file.txt"

    Convert instances to `str` in order to access these lines.
    """

    name: str
    typ: VarType
    value: str

    def __str__(self):
        """return the complete LAMMPS line"""
        if self.typ is VarType.STRING:
            return f"variable {self.name} {self.typ.value} \"{self.value}\"\n"
        return f"variable {self.name} {self.typ.value} {self.value}\n"


class LammpsRunner:
    """runs generic LAMMPS scripts after making the variable injections"""
    generic_lmp: str
    required_vars: List[LammpsVarLine]

    def __init__(self, generic_lmp, required_vars):
        generic_lmp = os.path.expanduser(generic_lmp)
        if os.path.isfile(generic_lmp):
            self.generic_lmp = generic_lmp
        else:
            raise FileNotFoundError(f"'{generic_lmp}' does not exist")
        self.required_vars = required_vars
        self._validate_provided_vars()
        self._commands_list = None

    @property
    def commands_list(self):
        if self._commands_list is None:
            self._commands_list = self._generate_commands_list()
        return self._commands_list

    def run(self,
            append_commands: List[str] = None,
            skip_lines: List[int] = None,
            skip_keywords: List[str] = None) -> None:
        """inject variables and run the resulting script"""
        if append_commands is None:
            append_commands = []

        lmp = MyLammps()
        commands_list = self._get_skipped_commands_list(skip_lines, skip_keywords)
        commands_list += append_commands

        self._commands_list = commands_list
        lmp.commands_list(self._commands_list)
        lmp.close()

    def write_final_script(self, filename: str = None) -> str:
        """write copy of the final script"""
        if filename is None:
            filename = f"in_{MyPhonolammps.get_random_id_string(12)}.lammps"

        with open(filename, 'w') as _file:
            _file.write("".join(self.commands_list))

        return filename

    def _generate_commands_list(self) -> List[str]:
        """generate list of commands to pass to `lammps.lammps.commands_list`"""
        commands_list = [str(v_line) for v_line in self.required_vars]
        with open(self.generic_lmp) as _file:
            commands_list += _file.readlines()
        return commands_list

    def _validate_provided_vars(self) -> None:
        """check that required vars contains all undefined variables"""
        provided_var_names = {v.name for v in self.required_vars}
        undefined_var_names = self._find_undefined_vars()
        var_set_diff = undefined_var_names - provided_var_names
        if len(var_set_diff) > 0:
            msg = "missing variables: "
            for var_name in var_set_diff:
                msg += f"'{var_name}', "
            msg += f"for input file '{self.generic_lmp}'"
            raise MissingVarError(msg)

    def _find_undefined_vars(self) -> Set[str]:
        """find undefined variables in the generic lammps script"""
        with open(self.generic_lmp) as _file:
            ss = "".join(_file.readlines())

        var_decs = re.findall("^variable.*", ss, flags=re.MULTILINE)
        declared = {}
        for var_dec in var_decs:
            name = var_dec.split()[1].strip()
            declared[name] = None

        var_evaluations = []
        for s in ss:
            if not s.startswith('#'):
                var_evals = re.findall(r"(\${\w+}|\$\w+)", s)
                for var_eval in var_evals:
                    var_evaluations.append(var_eval.strip("${}"))

        return {v for v in var_evaluations if v not in declared}

    def _get_skipped_commands_list(self,
                                   skip_lines: List[int],
                                   skip_keywords: List[str]) -> List[str]:
        """selectively delete certain commands from the input file"""
        # first, exclude any lines indexed in `skip_lines`
        raw_commands_list = self.commands_list
        commands_list = []
        skip_lines = [] if skip_lines is None else skip_lines
        for i, command in enumerate(raw_commands_list):
            if i not in skip_lines:
                commands_list.append(command)

        # exclude any lines starting with `skip_keywords` items
        raw_commands_list = commands_list
        commands_list = []
        skip_keywords = [] if skip_keywords is None else skip_keywords
        skip_keywords = [kw + " " for kw in skip_keywords
                         if not (kw.endswith(' ') or kw.endswith('\n'))]

        for command in raw_commands_list:
            match = False
            for kw in skip_keywords:
                if command.startswith(kw):
                    match = True
                    break
            if not match:
                commands_list.append(command)

        return commands_list
