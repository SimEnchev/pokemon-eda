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
| #   | Column             | Non-Null Count | Dtype   |
| --- | ------------------ | -------------- | ------- |
| 0   | abilities          | 801            | object  |
| 1   | against_bug        | 801            | float64 |
| 2   | against_dark       | 801            | float64 |
| 3   | against_dragon     | 801            | float64 |
| 4   | against_electric   | 801            | float64 |
| 5   | against_fairy      | 801            | float64 |
| 6   | against_fight      | 801            | float64 |
| 7   | against_fire       | 801            | float64 |
| 8   | against_flying     | 801            | float64 |
| 9   | against_ghost      | 801            | float64 |
| 10  | against_grass      | 801            | float64 |
| 11  | against_ground     | 801            | float64 |
| 12  | against_ice        | 801            | float64 |
| 13  | against_normal     | 801            | float64 |
| 14  | against_poison     | 801            | float64 |
| 15  | against_psychic    | 801            | float64 |
| 16  | against_rock       | 801            | float64 |
| 17  | against_steel      | 801            | float64 |
| 18  | against_water      | 801            | float64 |
| 19  | attack             | 801            | int64   |
| 20  | base_egg_steps     | 801            | int64   |
| 21  | base_happiness     | 801            | int64   |
| 22  | base_total         | 801            | int64   |
| 23  | capture_rate       | 801            | object  |
| 24  | classfication      | 801            | object  |
| 25  | defense            | 801            | int64   |
| 26  | experience_growth  | 801            | int64   |
| 27  | height_m           | 781            | float64 |
| 28  | hp                 | 801            | int64   |
| 29  | japanese_name      | 801            | object  |
| 30  | name               | 801            | object  |
| 31  | percentage_male    | 703            | float64 |
| 32  | sp_attack          | 801            | int64   |
| 33  | sp_defense         | 801            | int64   |
| 34  | speed              | 801            | int64   |
| 35  | type1              | 801            | object  |
| 36  | type2              | 417            | object  |
| 37  | weight_kg          | 781            | float64 |
| 38  | generation         | 801            | int64   |
| 39  | is_legendary       | 801            | int64   |
<br />
<br />
The biggest issue with this table is that the first column of the DataFrame is not the name of a Pokemon, but its abilities. While not entirely necessary, setting the first column to be the name of a Pokemon allows for much better readability. In addition, the Japanese names of the Pokemon can also be displayed next to the English name. The following code addresses this:

<br />
<br />

```
poke_name = poke_data.pop('name')
poke_data.insert(0,'name',poke_name)
poke_data.insert(1,'japanese_name',poke_data.pop('japanese_name'))
poke_data.head(20)
```
