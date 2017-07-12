####################################
# This file was created by Bohrium.
# It allows you to run NumPy code (cells) as Bohrium, by using the magic command
# `%%bohrium` in your cells, e.g.:
#
#    %%bohrium
#    print(numpy)
#    print(numpy.arange(10))
####################################
from IPython.core.magic import register_cell_magic

try:
    import bohrium

    have_bohrium = True

    @bohrium.replace_numpy
    def execute(__code):
        exec(__code, globals(), locals())
        __excludes = set(["__excludes", "__code", "np", "bohrium"])

        try:
            # Python 2.x
            for key, value in locals().iteritems():
                if key not in __excludes:
                    globals()[key] = value
        except:
            # Python 3.x
            for key, value in locals().items():
                if key not in __excludes:
                    globals()[key] = value
except ImportError:
    warning_shown = False # Warning about missin bohrium has been shown

    def execute(__code):
        global warning_shown

        if not warning_shown:
            print("WARNING: Module bohrium could not be imported.\n"
                  "         The magic command '%%bohrium' will have no effect.")
            warning_shown = True

        exec(__code, globals())


@register_cell_magic
def bohrium(line, cell):
    # Code must end with \n
    code = cell if cell.endswith("\n") else cell + "\n"
    execute(code)
    return
