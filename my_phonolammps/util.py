"""some basic utilities"""


def _print(s: str) -> None:
    """a silly custom debugging printer"""
    s_ = "\n\n---> my_phonolammps-debug\n"
    print(s_ + s + "\n\n")
