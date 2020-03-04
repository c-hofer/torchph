import sys
import os


def check_torchph_availability():
    try:
        import torchph

    except ImportError:
        sys.path.append(os.path.dirname(os.getcwd()))

        try:
            import torchph

        except ImportError as ex:
            raise ImportError(
                """
                Could not import torchph. Running your python \ 
                interpreter in the 'tutorials' sub folder could resolve \
                this issue.
                """
            ) from ex
