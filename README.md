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
# PART II - Descriptive Statistics and Data Cleanup
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

The biggest issue with this table is that the first column of the DataFrame is not the name of a Pokemon, but its abilities. While not entirely necessary, setting the first column to be the name of a Pokemon allows for much better readability. In addition, the Japanese names of the Pokemon can also be displayed next to the English name. The following code addresses this:
<br />

```
poke_name = poke_data.pop('name')
poke_data.insert(0,'name',poke_name)

poke_data.insert(1,'japanese_name',poke_data.pop('japanese_name'))
poke_data.head(11)
```
|   | name       | japanese_name       | abilities                    | against_bug | against_dark | against_dragon | against_electric | against_fairy | against_fight | against_fire | ... | pokedex_number | sp_attack | sp_defense | speed | type1 | type2 | weight_kg | generation | is_legendary | total_power |
|---|------------|---------------------|------------------------------|-------------|--------------|----------------|------------------|----------------|---------------|--------------|-----|----------------|-----------|-------------|-------|-------|-------|-----------|------------|--------------|-------------|
| 0 | Bulbasaur  | Fushigidaneフシギダネ  | ['Overgrow', 'Chlorophyll'] | 1.00        | 1.0          | 1.0            | 0.5              | 0.5            | 0.5           | 2.0          | ... | 1              | 65        | 65          | 45    | grass | poison | 6.9       | 1          | 0            | 318         |
| 1 | Ivysaur    | Fushigisouフシギソウ   | ['Overgrow', 'Chlorophyll'] | 1.00        | 1.0          | 1.0            | 0.5              | 0.5            | 0.5           | 2.0          | ... | 2              | 80        | 80          | 60    | grass | poison | 13.0      | 1          | 0            | 405         |
| 2 | Venusaur   | Fushigibanaフシギバナ   | ['Overgrow', 'Chlorophyll'] | 1.00        | 1.0          | 1.0            | 0.5              | 0.5            | 0.5           | 2.0          | ... | 3              | 122       | 120         | 80    | grass | poison | 100.0     | 1          | 0            | 625         |
| 3 | Charmander | Hitokageヒトカゲ       | ['Blaze', 'Solar Power']    | 0.50        | 1.0          | 1.0            | 1.0              | 0.5            | 1.0           | 0.5          | ... | 4              | 60        | 50          | 65    | fire  | NaN    | 8.5       | 1          | 0            | 309         |
| 4 | Charmeleon | Lizardoリザード        | ['Blaze', 'Solar Power']    | 0.50        | 1.0          | 1.0            | 1.0              | 0.5            | 1.0           | 0.5          | ... | 5              | 80        | 65          | 80    | fire  | NaN    | 19.0      | 1          | 0            | 405         |
| 5 | Charizard  | Lizardonリザードン     | ['Blaze', 'Solar Power']    | 0.25        | 1.0          | 1.0            | 2.0              | 0.5            | 0.5           | 0.5          | ... | 6              | 159       | 115         | 100   | fire  | flying | 90.5      | 1          | 0            | 634         |
| 6 | Squirtle   | Zenigameゼニガメ       | ['Torrent', 'Rain Dish']    | 1.00        | 1.0          | 1.0            | 2.0              | 1.0            | 1.0           | 0.5          | ... | 7              | 50        | 64          | 43    | water | NaN    | 9.0       | 1          | 0            | 314         |
| 7 | Wartortle  | Kameilカメール         | ['Torrent', 'Rain Dish']    | 1.00        | 1.0          | 1.0            | 2.0              | 1.0            | 1.0           | 0.5          | ... | 8              | 65        | 80          | 58    | water | NaN    | 22.5      | 1          | 0            | 405         |
| 8 | Blastoise  | Kamexカメックス         | ['Torrent', 'Rain Dish']    | 1.00        | 1.0          | 1.0            | 2.0              | 1.0            | 1.0           | 0.5          | ... | 9              | 135       | 115         | 78    | water | NaN    | 85.5      | 1          | 0            | 630         |
| 9 | Caterpie   | Caterpieキャタピー       | ['Shield Dust', 'Run Away'] | 1.00        | 1.0          | 1.0            | 1.0              | 1.0            | 0.5           | 2.0          | ... | 10             | 20        | 20          | 45    | bug   | NaN    | 2.9       | 1          | 0            | 195         |
|10 | Metapod    | Transelトランセル       | ['Shed Skin']               | 1.00        | 1.0          | 1.0            | 1.0              | 1.0            | 0.5           | 2.0          | ... | 11             | 25        | 25          | 30    | bug   | NaN    | 9.9       | 1          | 0            | 205         |


```
poke_data.tail(5)
```
|    | name       | japanese_name   | abilities           | against_bug   | against_dark   | against_dragon   | against_electric   | against_fairy   | against_fight   | against_fire   | ... | pokedex_number   | sp_attack   | sp_defense   | speed   | type1     | type2   | weight_kg   | generation   | is_legendary   | total_power   |
|----|------------|-----------------|---------------------|---------------|----------------|------------------|--------------------|-----------------|-----------------|----------------|-----|-----------------|-------------|--------------|---------|-----------|---------|-------------|--------------|----------------|---------------|
| 796| Celesteela | Tekkaguyaテッカグヤ | ['Beast Boost']     | 0.25          | 1.0            | 0.5              | 2.0                | 0.5             | 1.0             | 2.0            | ... | 797             | 107         | 101          | 61      | steel     | flying  | 999.9       | 7            | 1              | 570           |
| 797| Kartana    | Kamiturugiカミツルギ | ['Beast Boost']     | 1.0           | 1.0            | 0.5              | 0.5                | 0.5             | 2.0             | 4.0            | ... | 798             | 59          | 31           | 109     | grass     | steel   | 0.1         | 7            | 1              | 570           |
| 798| Guzzlord   | Akuzikingアクジキング | ['Beast Boost']     | 2.0           | 0.5            | 2.0              | 0.5                | 4.0             | 2.0             | 0.5            | ... | 799             | 97          | 53           | 43      | dark      | dragon  | 888.0       | 7            | 1              | 570           |
| 799| Necrozma   | Necrozmaネクロズマ | ['Prism Armor']      | 2.0           | 2.0            | 1.0              | 1.0                | 1.0             | 0.5             | 1.0            | ... | 800             | 127         | 89           | 79      | psychic   | NaN     | 230.0       | 7            | 1              | 600           |
| 800| Magearna   | Magearnaマギアナ | ['Soul-Heart']       | 0.25          | 0.5            | 0.0              | 1.0                | 0.5             | 1.0             | 2.0            | ... | 801             | 130         | 115          | 65      | steel     | fairy   | 80.5        | 7            | 1              | 600           |
<br />
Next, we can get an insight on the 'Base Stats' of Pokemon. The Base Stats consist of the 'hp', 'attack', 'defense', 'sp_attack', 'sp_defense', and speed of Pokemon.

```
base_stats = poke_data[['hp','attack','defense','sp_attack','sp_defense','speed']]
base_stats.describe()
```

|          |    hp |   attack |   defense |   sp_attack |   sp_defense |   speed |
|----------|-------|----------|-----------|-------------|--------------|---------|
| count    | 801.0 |    801.0 |     801.0 |       801.0 |        801.0 |   801.0 |
| mean     |  68.96|     77.86|      73.01|        71.31|         70.91|    66.33|
| std      |  26.58|     32.16|      30.77|        32.35|         27.94|    28.91|
| min      |   1.0 |      5.0 |       5.0 |        10.0 |         20.0 |     5.0 |
| 25%      |  50.0 |     55.0 |      50.0 |        45.0 |         50.0 |    45.0 |
| 50%      |  65.0 |     75.0 |      70.0 |        65.0 |         66.0 |    65.0 |
| 75%      |  80.0 |    100.0 |      90.0 |        91.0 |         90.0 |    85.0 |
| max      | 255.0 |    185.0 |     230.0 |       194.0 |        230.0 |   180.0 |

<br />
We can also do the same for the continous data such as height, weight, capture rate, base egg steps, experience growth, and base happiness.

```
cont_data = cont_data[['height_m', 'weight_kg', 'capture_rate', 'base_egg_steps', 'experience_growth', 'base_happiness']]
cont_data.describe()
```

|         | height_m   | weight_kg   | base_egg_steps   | experience_growth   | base_happiness   |
|---------|------------|-------------|-----------------|---------------------|------------------|
| count   | 781.000000 | 781.000000  | 801.000000      | 8.010000e+02        | 801.000000       |
| mean    | 1.163892   | 61.378105   | 7191.011236     | 1.054996e+06        | 65.362047        |
| std     | 1.080326   | 109.354766  | 6558.220422     | 1.602558e+05        | 19.598948        |
| min     | 0.100000   | 0.100000    | 1280.000000     | 6.000000e+05        | 0.000000         |
| 25%     | 0.600000   | 9.000000    | 5120.000000     | 1.000000e+06        | 70.000000        |
| 50%     | 1.000000   | 27.300000   | 5120.000000     | 1.000000e+06        | 70.000000        |
| 75%     | 1.500000   | 64.800000   | 6400.000000     | 1.059860e+06        | 70.000000        |
| max     | 14.500000  | 999.900000  | 30720.000000    | 1.640000e+06        | 140.000000       |

