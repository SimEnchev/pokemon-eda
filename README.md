# Exploratory Data Analysis on Pokémon in Python
Calling all Pokémon trainers and data enthusiasts! Prepare yourself for an exhilarating adventure as we delve into the fascinating world of Pokémon through an exploratory data analysis project. Together, we will embark on a quest to discover the strongest Pokémon, unravel the intricate web of correlations between their base stats, explore the captivating distribution of different Pokémon types, and decipher the unique strengths and weaknesses of individual Pokémon through radar charts. But our journey doesn't stop there! We will also unveil the average power wielded by each Pokémon type, determining which types reign supreme in battle. Additionally, we will investigate the capture rates of various Pokémon types, seeking to uncover any patterns that link rarity and capturability.

So, gather your Poké Balls and sharpen your analytical skills, for together, we will embark on a grand adventure, capturing the essence of Pokémon power and revealing the wonders concealed within our dataset. Are you prepared to join the ranks of the elite Pokémon researchers? Let's embark on this thrilling journey and catch 'em all!

# PART I - Importing and Reading the Dataset
```
import math
import numpy as np
import pandas as pd 
import matplotlib  
pd.plotting.register_matplotlib_converters()
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
```
```
filepath = "pokemon.csv"

poke_data = pd.read_csv(filepath, 
                           index_col = 'pokedex_number')
```
# PART II - Descriptive Statistics
```
poke_data.shape
```
(801, 42)
```
poke_data.info()
```

| Column             | Non-Null Count | Dtype   |
| ------------------ | -------------- | ------- |
| abilities          | 801            | object  |
| against_bug        | 801            | float64 |
| against_dark       | 801            | float64 |
| against_dragon     | 801            | float64 |
| against_electric   | 801            | float64 |
| against_fairy      | 801            | float64 |
| against_fight      | 801            | float64 |
| against_fire       | 801            | float64 |
| against_flying     | 801            | float64 |
| against_ghost      | 801            | float64 |
| against_grass      | 801            | float64 |
| against_ground     | 801            | float64 |
| against_ice        | 801            | float64 |
| against_normal     | 801            | float64 |
| against_poison     | 801            | float64 |
| against_psychic    | 801            | float64 |
| against_rock       | 801            | float64 |
| against_steel      | 801            | float64 |
| against_water      | 801            | float64 |
| attack             | 801            | int64   |
| base_egg_steps     | 801            | int64   |
| base_happiness     | 801            | int64   |
| base_total         | 801            | int64   |
| capture_rate       | 801            | object  |
| classfication      | 801            | object  |
| defense            | 801            | int64   |
| experience_growth  | 801            | int64   |
| height_m           | 781            | float64 |
| hp                 | 801            | int64   |
| japanese_name      | 801            | object  |
| name               | 801            | object  |
| percentage_male    | 703            | float64 |
| sp_attack          | 801            | int64   |
| sp_defense         | 801            | int64   |
| speed              | 801            | int64   |
| type1              | 801            | object  |
| type2              | 417            | object  |
| weight_kg          | 781            | float64 |
| generation         | 801            | int64   |
| is_legendary       | 801            | int64   |
