def info(df,name=''):
  print('============================DIMENSIONES===================================')
  print('Dataframe {:s}  --->  '.format(name),df.shape)
  print('============================INFORMACION===================================')
  print(df.info())
  print('============================COLUMNAS======================================')
  na_columns = df[df.columns[df.isna().any()]].isnull()
  print('Columnas con valores NA en porcentaje')
  print(100*na_columns.sum()/(len(df.index)))
  print('--------------------------------------------------------------------------')
  numeric_columns=df._get_numeric_data().columns
  print('Columnas numéricas  -> ',len(numeric_columns))
  print(numeric_columns)
  print('--------------------------------------------------------------------------')
  non_numeric_columns=df.select_dtypes(include='object').columns
  if (len(non_numeric_columns)>0):
    print('Columnas no numéricas  -> ',len(non_numeric_columns))
    print(non_numeric_columns)
  print('--------------------------------------------------------------------------')
  datetime_columns=df.select_dtypes(include='datetime64').columns
  if (len(datetime_columns)>0):
    print('Columnas datetime  -> ',len(datetime_columns))
    print(datetime_columns)
  print('===========================FILAS==========================================')
  rows_na = df[df.isnull().any(axis=1)].index
  print('filas con valores na o null --> ',len(rows_na))
  print('Representan el {:2.2f}%'.format(len(rows_na)*100/len(df)))
  print(rows_na)    
  return {'numeric_columns':numeric_columns,'non_numeric_columns':non_numeric_columns,'datetime_columns':datetime_columns,'na_columns':na_columns,'na_rows_index':rows_na}