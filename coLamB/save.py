# %%
import datetime
import os
import shutil


def create_directory(directory_name: str = "tmp", force: bool = False) -> None:
    """
    Creates a new directory with the specified name. If the directory already exists, it can be forcibly removed and recreated based on the `force` parameter.

    Args:
        directory_name (str, optional): The name of the directory to create. Default is "tmp".
        force (bool, optional): If True, the existing directory will be removed and recreated. Default is False.

    Returns:
        None
    """

    print(f"making <{directory_name}> directory...")
    if os.path.exists(directory_name):
        if force:
            shutil.rmtree(directory_name)
            print(f"Removed existing directory: {directory_name}")
        else:
            print(f"The directory <{directory_name}> already exists.")
            return

    os.makedirs(directory_name)
    print(f"Created a new directory: {directory_name}")


def make_dir_name(file_name: str = "output") -> str:
    """
    Generates a directory name by appending a timestamp to the specified base file name.

    Args:
        file_name (str, optional): The base name for the directory. Default is "output".

    Returns:
        str: The generated directory name with a timestamp.
    """
    current_time = datetime.datetime.now()
    timestamp = current_time.strftime("%Y-%m-%d-%H-%M-%S")
    filename = f"{file_name}_{timestamp}"
    return filename


# %%
