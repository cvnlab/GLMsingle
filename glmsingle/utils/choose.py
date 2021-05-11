"""
function f = choose(flag,yes,no)

 function f = choose(flag,yes,no)

 <flag> is a truth value (0 or 1)
 <yes> is something
 <no> is something

 if <flag>, return <yes>.  otherwise, return <no>.
"""


def choose(flag, yes, no):

    if flag:
        f = yes
    else:
        f = no
    return f
