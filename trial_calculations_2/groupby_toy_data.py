import pandas as pd

df = pd.DataFrame([['A boy ran.', [1,2], 1, [5,7], 0.997], ['A good pet.', [7,9], 0, [3,2], 0.977], ['The car is fast.', [7,5], 1, [1,9], 0.962], ['The girl sang.', [0,5], 2, [4,1], 0.992]], columns=['sentences', 'embeddings', 'labels', 'cluster_centres', 'cosine_scores'])
#print(df)

#Method 1 WORKS
# new_df = df.sort_values(['labels', 'cosine_scores'], ascending=False).drop_duplicates(['labels'])
# print(new_df)

#Method 2 WORKS
# idx = df.groupby(['labels'])['cosine_scores'].transform(max) == df['cosine_scores']
# print(df[idx])

#Method 3 WORKS
print(df.loc[df.groupby(["labels"])["cosine_scores"].idxmax()])
