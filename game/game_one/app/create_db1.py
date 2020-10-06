import sqlite3

connection = sqlite3.connect('database1.db')

with open('schema1.sql') as f:
    connection.executescript(f.read())

cur = connection.cursor()
