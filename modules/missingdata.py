import missingno as msno

def graph(df,fields=[]):
    msno.bar(df)
    msno.matrix(df.sort_values(by=fields))
    msno.heatmap(df)
    msno.dendrogram(df)
    
