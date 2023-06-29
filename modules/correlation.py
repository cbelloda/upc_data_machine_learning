def correlation(df,target='',size_corr=(30,15)):
  plt.figure(figsize=size_corr)

  corr = df.select_dtypes(include=np.number).corr()
  ax = sns.heatmap(
    corr,
    vmin=-1, vmax=1, center=0,
    cmap=sns.diverging_palette(20, 220, n=200),
    square=True,
    annot=True
  )
  ax.set_xticklabels(
    ax.get_xticklabels(),
    rotation=45,
    horizontalalignment='right'
  )
  plt.show()
  if target!='':
    print('Correlación con la variable objetivo')
    print(corr[target].sort_values(ascending=False))
  return corr
