# mysql_connector.py
import mysql.connector

def create_connection():
    try:
        connection = mysql.connector.connect(
            host='localhost',
            user='root',
            password='123456',  # Replace with your MySQL root password
            port='3306',
            database='mysql_pycharm'  # Replace with your database name
        )
        print("Connection to MySQL DB successful")
        return connection
    except mysql.connector.Error as e:
        print(f"The error '{e}' occurred")
        return None

def create_table(connection):
    try:
        cursor = connection.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS example_table (
                id INT AUTO_INCREMENT PRIMARY KEY,
                name VARCHAR(255) NOT NULL
            )
        """)

        print("Table 'example_table' created successfully")
    except mysql.connector.Error as e:
        print(f"The error '{e}' occurred")

# Test connection and create table
if __name__ == "__main__":
    conn = create_connection()
    if conn:
        create_table(conn)
