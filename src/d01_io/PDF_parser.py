import camelot
from pathlib import Path

# script to extract data from tables in PDFs

def shuffle_columns(df):
    df = df.drop([0, 1, 2, 3])  # remove header of table

    # shift multicolumn format to single column
    cols1 = df[[0, 1, 2, 3]]
    cols2 = df[[4, 5, 6, 7]].T.reset_index(drop=True).T
    cols3 = df[[8, 9, 10, 11]].T.reset_index(drop=True).T

    df = cols1.copy()
    df = df.append(cols2)
    df = df.append(cols3).set_index(0)

    df = df.rename(columns={1: "Residue 1", 2: "Residue 2", 3: "Upper bound [nm]"})
    return df


def split_and_match(df):
    # split into two columns
    df[["Residue_index_1", "Residue_name_1"]] = df["Residue 1"].str.split(" ", 1, expand=True)
    df[["Residue_index_2", "Residue_name_2"]] = df["Residue 2"].str.split(" ", 1, expand=True)
    # drop unsplit columns
    df = df.drop(labels=["Residue 1", "Residue 2"], axis=1)
    # reorder
    cols = df.columns.tolist()
    cols = cols[1:] + [cols[0]]
    df = df[cols]
    # changes to match naming convention
    df = df.rename(columns={"Upper bound [nm]": "Upper_bound_[nm]"})
    df.index.name = "Index"
    df = df.dropna()
    return df


'''
if Path('../../data/10_DecaPep/00_Decapeptides_SI.pdf').exists():
    print("ok")
else:
    print("not ok")
'''

tables = camelot.read_pdf('../../data/10_DecaPep/00_Decapeptides_SI.pdf', pages="6,7,8,9,10", flavor='stream')

# check data
data = []
for i in range(tables.n):
    print(i)
    data.append(tables[i].df)
    print(tables[i].parsing_report)


####################################################################
# DPep1,3,4,5,6                                                    #
####################################################################
dpep1 = split_and_match(shuffle_columns(data[0]))
dpep3 = split_and_match(shuffle_columns(data[2]))
dpep4 = split_and_match(shuffle_columns(data[3]))
dpep5 = split_and_match(shuffle_columns(data[4]))
dpep6 = split_and_match(shuffle_columns(data[5]))

# write out
dpep1.to_csv("../../data/11_DPep1/10_DPep1_chcl3_noebounds.csv", index=True, header=True, sep=" ")
dpep3.to_csv("../../data/13_DPep3/10_DPep3_chcl3_noebounds.csv", index=True, header=True, sep=" ")
dpep4.to_csv("../../data/14_DPep4/10_DPep4_chcl3_noebounds.csv", index=True, header=True, sep=" ")
dpep5.to_csv("../../data/15_DPep5/10_DPep5_chcl3_noebounds.csv", index=True, header=True, sep=" ")
dpep6.to_csv("../../data/16_DPep6/10_DPep6_chcl3_noebounds.csv", index=True, header=True, sep=" ")


####################################################################
# DPep2                                                            #
####################################################################
dpep2 = data[1].copy()
dpep2 = dpep2.drop([0]) #remove header of table
dpep2 = dpep2.set_index(0)
# move column names to header
dpep2 = dpep2.rename(columns=dpep2.iloc[0])
dpep2 = dpep2.drop(["Index"])

dpep2 = split_and_match(dpep2)

# write out
dpep2.to_csv("../../data/12_DPep2/10_DPep2_chcl3_noebounds.csv", index=True, header=True, sep=" ")


print("end")