import pandas as pd

# 读取两个表
df1 = pd.read_excel("pingfen.xlsx")   # 包含“学号”和“姓名”
df2 = pd.read_excel("合并成绩结果.xlsx")   # 包含“学号”和“姓名”
df3 = pd.read_excel("qiandao.xlsx", header=1)
# 清洗列名、学号列转字符串
for df in [df1, df2, df3]:
    df.columns = df.columns.str.strip()
    df["学号"] = df["学号"].astype(str)

# 只保留“学号”和“姓名”
df1_sub = df1[["学号", "真实姓名"]].drop_duplicates()
df2_sub = df2[["学号", "姓名"]].drop_duplicates()
df3_sub = df3[["学号", "姓名"]].drop_duplicates()
# 合并两个表，标记来源
df1_sub["来源"] = "表1"
df2_sub["来源"] = "表2"
df3_sub["来源"] = "表3"

df_all = pd.concat([df3_sub, df2_sub], ignore_index=True)

# 统计每个学号出现的次数
counts = df_all.groupby("学号").size().reset_index(name="出现次数")

# 找出只出现一次的学号
only_once_ids = counts[counts["出现次数"] == 1]["学号"]

# 根据学号筛选原始表，查看是在哪个表
only_once_students = df_all[df_all["学号"].isin(only_once_ids)]

# 输出结果
print("✅ 只出现在一个表中的学生如下：")
print(only_once_students[["学号", "姓名", "来源"]])
