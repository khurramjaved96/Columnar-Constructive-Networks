import subprocess
import mysql.connector
import sys
import time
import json

with open("credentials.json") as f:
    db_data = json.load(f)


def del_query(database, query):
    db_name = database
    conn = mysql.connector.connect(
        host=db_data['database'][0]["ip"],
        user=db_data['database'][0]["username"],
        password=db_data['database'][0]["password"])

    sql_run = conn.cursor()
    sql_run.execute("USE " + db_name + ";")
    output = sql_run.execute(query)
    conn.commit()
    conn.close()
    return output

def run_query(database, query):
    db_name = database
    print(query)
    conn = mysql.connector.connect(
        host=db_data['database'][0]["ip"],
        user=db_data['database'][0]["username"],
        password=db_data['database'][0]["password"])

    sql_run = conn.cursor()
    sql_run.execute("USE " + db_name + ";")
    output = sql_run.execute(query)
    output = sql_run.fetchall()
    conn.close()
    return output

p = subprocess.Popen('sleep 4', shell=True)
list_of_process = []
print("Running")
for i in range(62):
    list_of_process.append(subprocess.Popen('sleep 0.1', shell=True))
    time.sleep(0.2)
while True:
    try:
        counter = 0
        for p in list_of_process:
            if(p.poll() != None):
                output = run_query("experiment_queue", "select id, command from queue order by priority, rand() LIMIT 1;")
                if len(output) == 0:
                    print("Empty queue. Sleeping for 10 seconds\n")
                    time.sleep(10)
                for d in output:
                    print(d)
                    command = d[1]
                    del_query("experiment_queue", "delete from queue where id = " + str(d[0]) + ";")
                    print("DB command", command)
                    list_of_process[counter] = subprocess.Popen(command, shell=True)
                time.sleep(0.2)
            counter+=1

        time.sleep(3.0)
    except Exception as e:
        print("Exception ", e)
        time.sleep(10.0)