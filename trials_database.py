import sqlite3
import time

""" Select available studies in the database as well as trials` information and results """

if __name__ == '__main__':
    start = time.perf_counter()
    conn = sqlite3.connect('logs/cnn19_Oct_2021_16_28_23/cnn_study_19_Oct_2021_16_28_23.db')
    c = conn.cursor()

    # Print the name of the tables in the database
    print("Table names:")
    print(c.execute("SELECT Name FROM sqlite_master where type='table'").fetchall())

    # Select studies conducted
    print("Studies:")
    print(c.execute("SELECT * FROM studies").fetchall())

    # Select trial results
    print("Trial results:")
    print(c.execute("SELECT * FROM trials").fetchall())
    print(c.execute("SELECT * FROM trial_params").fetchall())
    print(c.execute("SELECT * FROM trial_values").fetchall())
    print(c.execute("SELECT * FROM trial_intermediate_values").fetchall())

    # Execution Time
    end = time.perf_counter()
    print(f"\nExecution time = {end - start:.2f} second(s)")
