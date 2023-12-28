import pandas as pd


df = pd.read_excel('Data/Dissertation Training Data.xlsx')
composers_grouped = df.groupby('Composer')['Composition'].apply(list).to_dict()

output_lines = []
for composer, compositions in composers_grouped.items():
    output_lines.append(composer)
    output_lines.extend(['\t' + composition for composition in compositions])
    output_lines.append('')

formatted_output = '\n'.join(output_lines)

print(formatted_output)
with open('test.txt', 'wb') as f:
  f.write(formatted_output.encode('utf-8'))
