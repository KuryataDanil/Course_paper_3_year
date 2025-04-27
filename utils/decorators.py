import sys
import time
from functools import wraps
from io import StringIO
import itertools
from threading import Thread, Event


def execution_time(description=None):
    """Декоратор для вывода времени выполнения (без перехвата вывода)"""

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            result = func(*args, **kwargs)

            elapsed = time.time() - start_time
            mins, secs = divmod(int(elapsed), 60)
            print(f"Время выполнения {description or func.__name__} - {mins:02d}:{secs:02d}\n")

            return result

        return wrapper

    return decorator

def execution_animation(description=None):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            nonlocal description
            description = description or func.__name__
            stop_event = Event()
            spinner_chars = itertools.cycle(['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏'])

            original_stdout = sys.stdout
            sys.stdout = StringIO()

            def spin():
                while not stop_event.is_set():

                    original_stdout.write(f"\r{description} {next(spinner_chars)}")
                    original_stdout.flush()
                    time.sleep(0.1)

            spinner_thread = Thread(target=spin)
            spinner_thread.start()

            try:
                result = func(*args, **kwargs)
            finally:
                stop_event.set()
                spinner_thread.join()

                func_output = sys.stdout.getvalue()
                sys.stdout = original_stdout
                if func_output:
                    print(f"\n{func_output}", end="")

                print(f"\r{description} ✓ Готово!")

            return result

        return wrapper

    return decorator
