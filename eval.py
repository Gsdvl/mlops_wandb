from fastai.vision.all import *
import pandas as pd

# Caminho para o arquivo CSV e a pasta de imagens
csv_path = 'data_split.csv'
path = Path('flowers')  # Este é o diretório base, que já contém as subpastas com imagens

# Carregar o CSV com as informações das imagens e suas classes
df = pd.read_csv(csv_path)

# Definir as colunas de classe
label_cols = ['daisy', 'dandelion', 'tulip', 'sunflower', 'rose']

# Criar uma lista de paths de imagens
# Corrigimos aqui, para não adicionar o diretório 'flowers' novamente
df['image_path'] = df['image'].apply(lambda x: path / x)

# Criar os DataLoaders a partir do DataFrame
dls = ImageDataLoaders.from_df(
    df, path, folder='', label_col=label_cols,  # Não usamos 'flowers' em 'folder'
    item_tfms=Resize(224), bs=32
)

# Visualizar um batch de dados
dls.show_batch()
