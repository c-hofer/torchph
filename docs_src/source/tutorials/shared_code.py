import sys
import os


def check_chofer_torchex_availability():
    try:
        import chofer_torchex

    except ImportError:
        sys.path.append(os.path.dirname(os.getcwd()))

        try:
            import chofer_torchex

        except ImportError as ex:
            raise ImportError(
                """
                Could not import chofer_torchex. Running your python \ 
                interpreter in the 'tutorials' sub folder could resolve \
                this issue.
                """
            ) from ex
