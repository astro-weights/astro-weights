save_peft now refuses a non-empty directory unless `overwrite=True`, and writes the adapter atomically through a temp directory.
