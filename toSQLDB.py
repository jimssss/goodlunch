import pymssql
from datetime import datetime, timedelta


#連接資料庫
server = 'jim0530.database.windows.net'
database = 'jim0530forLinebot'
username = 'adminjim'
password = 'TestTest0530'
conn = pymssql.connect(server, username, password, database)
cursor = conn.cursor(as_dict=True)
#查詢資料庫


current_date =datetime.now().date()
# current_date = (datetime.now() - timedelta(days=1)).date()
query = '''
SELECT 
    SUM(grain) AS total_grain,
    SUM(egg) AS total_egg,
    SUM(milk) AS total_milk,
    SUM(veg) AS total_veg,
    SUM(fruit) AS total_fruit,
    SUM(nuts) AS total_nuts,
    COUNT(*) AS total_records
FROM UserDietRecords
WHERE CAST(RecordDateTime AS DATE) = %s;
'''

# 执行查询
cursor.execute(query, (current_date))

# 获取查询结果
result = cursor.fetchone()

# 打印结果
if result:
    print(f"Grain: {result['total_grain']}")
    print(f"Egg: {result['total_egg']}")
    print(f"Milk: {result['total_milk']}")
    print(f"Vegetable: {result['total_veg']}")
    print(f"Fruit: {result['total_fruit']}")
    print(f"Nuts: {result['total_nuts']}")
else:
    print("No records found for the specified user and date.")

# 获取查询结果
# results = cursor.fetchall()

# # 打印结果
# for row in results:
#     print(row)

# 关闭连接
cursor.close()
conn.close()


# # 创建连接
# conn = pymssql.connect(server, username, password, database)

# # 创建游标对象
# cursor = conn.cursor()

# # 创建表

# cursor = conn.cursor()
# create_table_query = '''
# CREATE TABLE UserDietRecords (
#     LineID NVARCHAR(50) NOT NULL,
#     RecordDateTime DATETIME DEFAULT GETDATE(),
#     grain INT,
#     egg INT,
#     milk INT,
#     veg INT,
#     fruit INT,
#     nuts INT,
#     PRIMARY KEY (LineID, RecordDateTime)
# )
# '''



# cursor.execute(create_table_query)
# conn.commit()

# # 插入数据
# insert_query = '''
# INSERT INTO TestTable (ID, Name, Age)
# VALUES (%d, %s, %d)
# '''
# cursor.execute(insert_query, (1, 'John Doe', 30))
# conn.commit()

# data = [(2, 'Jane Smith', 25), (3, 'Mike Johnson', 35)]
# cursor.executemany(insert_query, data)
# conn.commit()

# # 读取数据
# select_query = 'SELECT * FROM TestTable'
# cursor.execute(select_query)
# rows = cursor.fetchall()
# for row in rows:
#     print(row)

# # 更新数据
# update_query = '''
# UPDATE TestTable
# SET Age = %d
# WHERE ID = %d
# '''
# cursor.execute(update_query, (31, 1))
# conn.commit()

# # 删除数据
# delete_query = '''
# DELETE FROM TestTable
# WHERE ID = %d
# '''
# cursor.execute(delete_query, (1,))
# conn.commit()

# 关闭连接
# cursor.close()
# conn.close()
