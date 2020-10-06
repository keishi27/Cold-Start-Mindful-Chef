# S2DS - Mindful Chef - August 2020

## Description
Team repo for Mindful Chef data science project, as part of Science to Data Science (S2DS) August 2020.

## Table of Contents

**data**: Store all csv files here. Analysis outputs are also generated here.

**flaskapp_andrius**: Wrapper and code to generate delta12.

**game_one**: Front and back-end code for Game 1 (baseline).

**game_three**: Front and back-end code for Game 3 (clustering).

**game_two**: Front and back-end code for Game 2 (extremes).

**notebooks**:
1. Exploratory data analysis.
2. Game 2 development: Dimensionality reduction of recipe table and generation of CSVs for Game 2. EDA into single customer journey in foodspace
3. Generate clean recipe table used in all games, fix image errors, set up analysis metrics and test calculations using sample customer database.
4. Analysis of data from A/B testing, comparison with historical customer data. Statistical analysis of results.
5. Clustering recipes using topic modelling and K-means, generating the csv file for Game 3.
6. Aditional attempts to cluster recipes.

**reports**: Updates and deliverables presented to S2DS and Mindful Chef.

**src**: Source code for project.

## Guide to Running Games

**Generate clean recipe table**
- Navigate to `game/src`
- Enter `python`, then enter `from feature_generator import clean_recipe_table`
- Enter `clean_recipe_table('../data/recipe_table.csv', '../data/recipe_table_updated_UAT.csv')`
- This generates the updated recipe table `../data/recipe_table_new.csv`
- Delete `../data/recipe_table.csv` (the old recipe table)
- Rename `../data/recipe_table_new.csv` to `../data/recipe_table.csv`

**Game 1**
- Navigate to `game/game_one/app`
- Enter `python app.py` to run the game

**Game 2**
- Navigate to `game/src`
- Enter `python`, then enter `from game2_builder import get_recipe_3_pc, get_scaled_pc_by_fg`
- Enter `get_recipe_3_pc('../data/recipe_table.csv')`
- This uses the updated recipe table to generate a new table with 3 principal components for each recipe
- Enter `get_scaled_pc_by_fg('../data/recipe_3_pc.csv')`
- This generates CSV files for each food group, where the each principal component axis is scaled between 0 and 1
- Example CSV file: `../data/Beef_FG_Scaled_PC.csv`
- Navigate to `game/game_two/app`
- Enter `python app.py` to run the game

**Game 3**
- Navigate to `game/src`
- Enter `python clustering_recipes.py` to generate the `.csv` file required by Game 3
- Navigate to `game/game_three/app`
- Enter `python app.py`
<br><br>*Caution:*
- When a new recipe table is used, it is necessary to navigate to game/notebooks and run the Jupyter Notebook : `5.0_clustering_recipes.ipynb`
- The user needs to try different parameters inside the topic modelling section to get the best number of topics without significant overlap and also simultaneously maximizing
- Any change made in the notebook has to be implemented in the python script: `src/clustering_recipies.py`, and then rerun to generate the required `.csv` file.

**Analysis of results**
- Navigate to `game/src`
- Enter `python`, then enter `from ab_test_analyser import analyse_abtest, analyse_abtest_game3`
<br><br>*For Game 1 and 2*
- Enter `analyse_abtest('existing_order.csv', 'observed_order.csv', game_number)` (with relevant csv names)
<br><br>*For Game 3*
- Enter `analyse_abtest_game3('existing_order.csv', 'observed_order.csv', game_number)` (with relevant csv names)
<br><br>*Then for all games*
- This generates the difference in customers' historical and measured values for delta12/magnitude, for each game
- Visualisation and further statistical analysis can be conducted by inputting the generated `.csv` files into `game/notebooks/4.3-statistical-analysis-results.ipynb`

## Installation 
1. Create a local virtual environment: 
- ubuntu/mac: `python3 -m venv venv`
- windows: `py -m venv venv`
2. Activate environment: 
- ubuntu/mac: `source venv/bin/activate`
- windows: `.\venv\scripts\activate`
3. Install requirements:
- run: `./terminal_installation.sh` file in the terminal

## Credits
Arya Dhar - arya.dhar@gmail.com

Giovanni De Cillis - giovanni.decillis@gmail.com

Keishi Kohara - keishi@live.co.uk

Lisa Richards - lisarichards81@gmail.com

Valentina Notararigo - valentina.notararigo@gmail.com
