f = open("db.ini", "r")
x, host = f.readline().split('=')
x, username = f.readline().split('=')
x, password = f.readline().split('=')
x, database = f.readline().split('=')

print(host)
print(username)
print(password)
print(database)