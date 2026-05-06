import pandas as pd
df = pd.read_csv('/workspaces/IMVmasterthesis/input_data/euromod_output/es_2017_std.txt', sep='\t', decimal=',')

galicia = df[df['drgn2'] == 11]
print('Persons in Galicia:', len(galicia))
print('bsarg_s non-zero in Galicia:', (galicia['bsarg_s'] > 0).sum())
print('bsarg_s mean in Galicia:', galicia[galicia['bsarg_s'] > 0]['bsarg_s'].mean())
print()

# For persons NOT in Galicia, do they get any bsarg_s?
not_galicia = df[df['drgn2'] != 11]
print('Persons outside Galicia:', len(not_galicia))
print('bsarg_s non-zero outside Galicia:', (not_galicia['bsarg_s'] > 0).sum())
print()

# Overall bsarg_s by region
print('bsarg_s recipients by region:')
print(df[df['bsarg_s'] > 0].groupby('drgn2')['bsarg_s'].agg(['count','mean']).round(2))

print('=== ils_earns (net earnings in means test) ===')
print(df['ils_earns'].describe())
print('Non-zero:', (df['ils_earns'] != 0).sum())
print()
print('=== ils_origy (original market income) ===')
print(df['ils_origy'].describe())
print('Non-zero:', (df['ils_origy'] != 0).sum())
print()
print('=== ils_ben (benefits in means test) ===')
print(df['ils_ben'].describe())
print('Non-zero:', (df['ils_ben'] != 0).sum())
print()

recipients = df[df['bsarg_s'] > 0]
non_recipients = df[df['bsarg_s'] == 0]
print('Mean ils_origy for recipients:', recipients['ils_origy'].mean().round(2))
print('Mean ils_origy for non-recipients:', non_recipients['ils_origy'].mean().round(2))
print()
print('Mean ils_earns for recipients:', recipients['ils_earns'].mean().round(2))
print('Mean ils_earns for non-recipients:', non_recipients['ils_earns'].mean().round(2))